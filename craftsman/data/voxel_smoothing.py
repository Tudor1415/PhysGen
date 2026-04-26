import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from craftsman import register
from craftsman.utils.config import parse_structured
from craftsman.utils.typing import *

from .objaverse import apply_transformation, random_mirror_matrix, random_rotation_matrix


@dataclass
class VoxelSmoothingDataModuleConfig:
    root_dir: str = None
    data_type: str = "sdf"
    load_supervision: bool = True
    supervision_type: str = "tsdf"
    n_supervision: list[int] = field(default_factory=lambda: [21384, 10000, 10000])
    rotate_points: bool = False
    batch_size: int = 32
    num_workers: int = 0
    train_variant_mode: str = "random"
    eval_variant_index: int = 0


class VoxelSmoothingDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: VoxelSmoothingDataModuleConfig = cfg
        self.split = split
        self.uids = self._load_uids(split)
        print(f"Loaded {len(self.uids)} {split} samples from {self.cfg.root_dir}")

    def _load_uids(self, split: str) -> list[str]:
        root = Path(self.cfg.root_dir)
        split_file = root / f"{split}.json"
        if not split_file.exists():
            raise RuntimeError(f"Missing split file: {split_file}")

        with split_file.open("r", encoding="utf-8") as handle:
            uids = json.load(handle)
        if not isinstance(uids, list):
            raise RuntimeError(f"{split_file} must contain a JSON list of file paths")

        resolved = []
        for uid in uids:
            uid_path = Path(uid)
            if not uid_path.is_absolute():
                uid_path = (root / uid_path).resolve()
            resolved.append(str(uid_path))
        return resolved

    def __len__(self):
        return len(self.uids)

    def _choice_indices(self, size: int, count: int, rng: np.random.Generator) -> np.ndarray:
        if size <= 0:
            raise RuntimeError("Cannot sample from an empty pool")
        return rng.choice(size, size=count, replace=size < count)

    def _select_variant_index(self, variant_count: int) -> int:
        if self.split == "train" and str(self.cfg.train_variant_mode).lower() == "random":
            rng = np.random.default_rng()
            return int(rng.integers(0, variant_count))

        mode = str(self.cfg.train_variant_mode)
        if self.split == "train":
            try:
                return int(mode) % variant_count
            except ValueError:
                pass
        return int(self.cfg.eval_variant_index) % variant_count

    def _load_variant_surface(self, data: np.lib.npyio.NpzFile, key: str, variant_idx: int) -> np.ndarray:
        surface = data[key][:, variant_idx, :]
        surface = np.nan_to_num(surface, nan=1.0, posinf=1.0, neginf=-1.0)
        return surface.astype(np.float32)

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
        return coarse_surface.astype(np.float32), sharp_surface.astype(np.float32), mirror_matrix, rotation_matrix

    def _load_shape_supervision(
        self,
        data: np.lib.npyio.NpzFile,
        mirror_matrix: Optional[np.ndarray],
        rotation_matrix: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        ret: Dict[str, Any] = {}
        sharp_near_surface = data["sharp_near_surface"]
        rand_points_with_sdf = data["rand_points"]

        sharp_near_points = sharp_near_surface[:, :3]
        sharp_sdfs = sharp_near_surface[:, 3]
        coarse_rand_points = rand_points_with_sdf[:, :3]
        coarse_sdfs = rand_points_with_sdf[:, 3]

        rand_points_near_count = int(data["rand_points_near_count"]) if "rand_points_near_count" in data.files else min(400000, coarse_rand_points.shape[0])
        rand_points_near_count = min(rand_points_near_count, coarse_rand_points.shape[0])

        rng = np.random.default_rng()
        ind_sharp = self._choice_indices(sharp_near_points.shape[0], self.cfg.n_supervision[0], rng)
        ind_near = self._choice_indices(rand_points_near_count, self.cfg.n_supervision[1], rng)
        coarse_space_points = coarse_rand_points[rand_points_near_count:]
        coarse_space_sdfs = coarse_sdfs[rand_points_near_count:]
        ind_space = self._choice_indices(coarse_space_points.shape[0], self.cfg.n_supervision[2], rng)

        rand_points = np.concatenate(
            [
                sharp_near_points[ind_sharp],
                coarse_rand_points[:rand_points_near_count][ind_near],
                coarse_space_points[ind_space],
            ],
            axis=0,
        )

        if mirror_matrix is not None and rotation_matrix is not None:
            mirrored_points, _ = apply_transformation(rand_points, None, mirror_matrix)
            rand_points, _ = apply_transformation(mirrored_points, None, rotation_matrix)

        ret["rand_points"] = rand_points.astype(np.float32)
        ret["number_sharp"] = int(self.cfg.n_supervision[0])

        if self.cfg.supervision_type == "occupancy":
            sdfs = np.concatenate(
                [
                    sharp_sdfs[ind_sharp],
                    coarse_sdfs[:rand_points_near_count][ind_near],
                    coarse_space_sdfs[ind_space],
                ],
                axis=0,
            )
            sdfs = np.nan_to_num(sdfs, nan=0.0, posinf=0.0, neginf=0.0)
            ret["occupancies"] = np.where(sdfs.flatten() < 0, 0, 1).astype(np.float32)
        elif self.cfg.supervision_type == "tsdf":
            sdfs = np.concatenate(
                [
                    sharp_sdfs[ind_sharp],
                    coarse_sdfs[:rand_points_near_count][ind_near],
                    coarse_space_sdfs[ind_space],
                ],
                axis=0,
            )
            sdfs = np.nan_to_num(sdfs, nan=0.0, posinf=0.0, neginf=0.0)
            ret["sdf"] = sdfs.flatten().astype(np.float32).clip(-0.015, 0.015) / 0.015
        else:
            raise NotImplementedError(f"Supervision type {self.cfg.supervision_type} not implemented")

        return ret

    def get_data(self, index):
        sample_path = self.uids[index]
        with np.load(sample_path, allow_pickle=False) as data:
            variant_count = int(data["fps_coarse_surface"].shape[1])
            variant_idx = self._select_variant_index(variant_count)

            coarse_surface = self._load_variant_surface(data, "fps_coarse_surface", variant_idx)
            sharp_surface = self._load_variant_surface(data, "fps_sharp_surface", variant_idx)

            mirror_matrix = None
            rotation_matrix = None
            if self.cfg.rotate_points and self.split == "train":
                coarse_surface, sharp_surface, mirror_matrix, rotation_matrix = self._apply_rotation(
                    coarse_surface, sharp_surface
                )

            ret = {
                "uid": sample_path,
                "coarse_surface": coarse_surface.astype(np.float32),
                "sharp_surface": sharp_surface.astype(np.float32),
            }

            if self.cfg.load_supervision:
                ret.update(self._load_shape_supervision(data, mirror_matrix, rotation_matrix))

        return ret

    def __getitem__(self, index):
        if self.split == "train":
            index %= len(self.uids)
        try:
            return self.get_data(index)
        except Exception as exc:
            print(f"Error in {self.uids[index]}: {exc}")
            if len(self) == 1:
                raise
            rng = np.random.default_rng()
            return self.__getitem__(int(rng.integers(0, len(self))))

    def collate(self, batch):
        return torch.utils.data.default_collate(batch)


@register("voxel-smoothing-datamodule")
class VoxelSmoothingDataModule(pl.LightningDataModule):
    cfg: VoxelSmoothingDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(VoxelSmoothingDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = VoxelSmoothingDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = VoxelSmoothingDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = VoxelSmoothingDataset(self.cfg, "test")

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
