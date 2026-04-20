import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import trimesh
from torch.utils.data import DataLoader, Dataset

from craftsman import register
from craftsman.utils.config import parse_structured
from craftsman.utils.typing import *

from .objaverse import apply_transformation, random_mirror_matrix, random_rotation_matrix

try:
    import fpsample
except Exception:
    fpsample = None


def _as_mesh(mesh_or_scene):
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene
    if isinstance(mesh_or_scene, trimesh.Scene):
        meshes = [
            geom
            for geom in mesh_or_scene.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0 and len(geom.faces) > 0
        ]
        if not meshes:
            raise RuntimeError("Scene does not contain any mesh geometry")
        return trimesh.util.concatenate(meshes)
    if isinstance(mesh_or_scene, (list, tuple)):
        meshes = [
            geom
            for geom in mesh_or_scene
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0 and len(geom.faces) > 0
        ]
        if not meshes:
            raise RuntimeError("Mesh list is empty")
        return trimesh.util.concatenate(meshes)
    raise TypeError(f"Unsupported mesh type: {type(mesh_or_scene)}")


@dataclass
class OnlineOBJDataModuleConfig:
    root_dir: str = None
    batch_size: int = 1
    num_workers: int = 0
    rotate_points: bool = False
    normalize_mesh: bool = True
    recursive: bool = True
    extensions: list[str] = field(default_factory=lambda: [".obj"])
    n_input_points: int = 32768
    coarse_pool_size: int = 200000
    sharp_angle_threshold_deg: float = 30.0


class OnlineOBJDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: OnlineOBJDataModuleConfig = cfg
        self.split = split
        self.uids = self._load_uids(split)
        print(f"Loaded {len(self.uids)} {split} meshes from {self.cfg.root_dir}")

    def _load_uids(self, split: str) -> list[str]:
        root = Path(self.cfg.root_dir)
        split_file = root / f"{split}.json"
        if split_file.exists():
            with open(split_file) as fp:
                uids = json.load(fp)
            if not isinstance(uids, list):
                raise RuntimeError(f"{split_file} must contain a JSON list of mesh paths")
            resolved = []
            for uid in uids:
                uid_path = Path(uid)
                if not uid_path.is_absolute():
                    uid_path = (root / uid_path).resolve()
                resolved.append(str(uid_path))
            return resolved

        patterns = []
        for ext in self.cfg.extensions:
            ext = ext if ext.startswith(".") else f".{ext}"
            patterns.append(f"*{ext}")

        paths = []
        for pattern in patterns:
            if self.cfg.recursive:
                paths.extend(root.rglob(pattern))
            else:
                paths.extend(root.glob(pattern))
        return [str(path.resolve()) for path in sorted(set(paths))]

    def __len__(self):
        return len(self.uids)

    def _normalize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        if vertices.size == 0:
            raise RuntimeError("Mesh has no vertices")
        bbmin = vertices.min(axis=0)
        bbmax = vertices.max(axis=0)
        center = 0.5 * (bbmin + bbmax)
        scale = 2.0 / max(float((bbmax - bbmin).max()), 1e-6)
        vertices = (vertices - center) * scale
        return trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.faces), process=False)

    def _load_mesh(self, path: str) -> trimesh.Trimesh:
        mesh = trimesh.load(path, force="mesh", process=False)
        mesh = _as_mesh(mesh)
        if self.cfg.normalize_mesh:
            mesh = self._normalize_mesh(mesh)
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        if len(mesh.faces) == 0:
            raise RuntimeError(f"Mesh has no faces: {path}")
        return mesh

    def _sample_surface_with_normals(self, mesh: trimesh.Trimesh, count: int) -> np.ndarray:
        points, face_ids = mesh.sample(count, return_index=True)
        normals = mesh.face_normals[face_ids]
        return np.concatenate([points, normals], axis=1).astype(np.float32)

    def _fps_downsample(self, samples: np.ndarray, target_count: int) -> np.ndarray:
        num_samples = samples.shape[0]
        if num_samples == target_count:
            return samples.astype(np.float32)
        if num_samples < target_count:
            rng = np.random.default_rng()
            indices = rng.choice(num_samples, size=target_count, replace=True)
            return samples[indices].astype(np.float32)
        if fpsample is not None:
            indices = fpsample.bucket_fps_kdline_sampling(samples[:, :3], target_count, h=5)
            return samples[np.asarray(indices, dtype=np.int64)].astype(np.float32)
        rng = np.random.default_rng()
        indices = rng.choice(num_samples, size=target_count, replace=False)
        return samples[indices].astype(np.float32)

    def _sample_sharp_surface(self, mesh: trimesh.Trimesh, target_count: int) -> np.ndarray:
        if len(mesh.face_adjacency_edges) == 0:
            return self._fps_downsample(
                self._sample_surface_with_normals(mesh, max(target_count, 4096)),
                target_count,
            )

        threshold = np.deg2rad(self.cfg.sharp_angle_threshold_deg)
        sharp_mask = mesh.face_adjacency_angles >= threshold

        if not np.any(sharp_mask):
            return self._fps_downsample(
                self._sample_surface_with_normals(mesh, max(target_count, 4096)),
                target_count,
            )

        sharp_edges = mesh.face_adjacency_edges[sharp_mask]
        sharp_faces = mesh.face_adjacency[sharp_mask]
        edge_vertices = mesh.vertices[sharp_edges]
        edge_lengths = np.linalg.norm(edge_vertices[:, 1] - edge_vertices[:, 0], axis=1)
        edge_lengths = np.clip(edge_lengths.astype(np.float64), 1e-12, None)
        probs = edge_lengths / edge_lengths.sum()

        normals = mesh.face_normals[sharp_faces].mean(axis=1)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.clip(norms, 1e-12, None)

        rng = np.random.default_rng()
        picked_edges = rng.choice(len(sharp_edges), size=target_count, replace=True, p=probs)
        t = rng.random((target_count, 1), dtype=np.float32)
        starts = edge_vertices[picked_edges, 0]
        ends = edge_vertices[picked_edges, 1]
        points = (1.0 - t) * starts + t * ends
        point_normals = normals[picked_edges]
        return np.concatenate([points, point_normals], axis=1).astype(np.float32)

    def _apply_rotation(self, coarse_surface: np.ndarray, sharp_surface: np.ndarray):
        mirror_matrix = random_mirror_matrix()
        rotation_matrix = random_rotation_matrix()

        coarse_points, coarse_normals = apply_transformation(
            coarse_surface[:, :3], coarse_surface[:, 3:], mirror_matrix
        )
        coarse_points, coarse_normals = apply_transformation(
            coarse_points, coarse_normals, rotation_matrix
        )

        sharp_points, sharp_normals = apply_transformation(
            sharp_surface[:, :3], sharp_surface[:, 3:], mirror_matrix
        )
        sharp_points, sharp_normals = apply_transformation(
            sharp_points, sharp_normals, rotation_matrix
        )

        coarse_surface = np.concatenate([coarse_points, coarse_normals], axis=1)
        sharp_surface = np.concatenate([sharp_points, sharp_normals], axis=1)
        return coarse_surface.astype(np.float32), sharp_surface.astype(np.float32)

    def __getitem__(self, index):
        mesh_path = self.uids[index]
        try:
            mesh = self._load_mesh(mesh_path)
            coarse_surface = self._sample_surface_with_normals(
                mesh, max(self.cfg.coarse_pool_size, self.cfg.n_input_points)
            )
            coarse_surface = self._fps_downsample(coarse_surface, self.cfg.n_input_points)
            sharp_surface = self._sample_sharp_surface(mesh, self.cfg.n_input_points)

            if self.cfg.rotate_points and self.split == "train":
                coarse_surface, sharp_surface = self._apply_rotation(coarse_surface, sharp_surface)

            return {
                "uid": mesh_path,
                "coarse_surface": coarse_surface.astype(np.float32),
                "sharp_surface": sharp_surface.astype(np.float32),
            }
        except Exception as exc:
            print(f"Error in {mesh_path}: {exc}")
            if len(self) == 1:
                raise
            rng = np.random.default_rng()
            return self.__getitem__(int(rng.integers(0, len(self))))

    def collate(self, batch):
        return torch.utils.data.default_collate(batch)


@register("online-obj-datamodule")
class OnlineOBJDataModule(pl.LightningDataModule):
    cfg: OnlineOBJDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(OnlineOBJDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = OnlineOBJDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = OnlineOBJDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = OnlineOBJDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None, num_workers=0) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset,
            batch_size=1,
            collate_fn=self.val_dataset.collate,
            num_workers=self.cfg.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset,
            batch_size=1,
            collate_fn=self.test_dataset.collate,
            num_workers=self.cfg.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset,
            batch_size=1,
            collate_fn=self.test_dataset.collate,
            num_workers=self.cfg.num_workers,
        )
