import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import trimesh
from pysdf import SDF
from tqdm import tqdm

try:
    import fpsample
except Exception:
    fpsample = None


DEFAULT_COARSE_RESOLUTIONS = [32, 40, 48, 56, 64]
DEFAULT_TOOTH_RESOLUTIONS = [96, 112, 128, 144, 160]
DEFAULT_SHARP_NOISE_SCALES = [0.001, 0.005, 0.007, 0.01]
DEFAULT_RAND_NOISE_SCALES = [0.001, 0.005]


def _default_data_root() -> str | None:
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        return None
    return str(Path(scratch) / "seg_data")


def _default_output_root() -> str | None:
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        return None
    return str(Path(scratch) / "seg_data_voxel_smoothing")


def _parse_int_list(raw: str) -> List[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


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


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    mesh = _as_mesh(mesh)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"Mesh has no geometry: {path}")
    mesh.fix_normals()
    return mesh


def _normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    bbmin = vertices.min(axis=0)
    bbmax = vertices.max(axis=0)
    center = 0.5 * (bbmin + bbmax)
    scale = 2.0 / max(float((bbmax - bbmin).max()), 1e-6)
    normalized_vertices = (vertices - center) * scale
    normalized_mesh = trimesh.Trimesh(
        vertices=normalized_vertices,
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )
    normalized_mesh.fix_normals()
    return normalized_mesh


def _polygon_signed_area(points_2d: np.ndarray) -> float:
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _point_in_triangle_2d(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    eps: float = 1e-12,
) -> bool:
    v0 = c - a
    v1 = b - a
    v2 = point - a

    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) <= eps:
        return False

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    return u >= -eps and v >= -eps and (u + v) <= 1.0 + eps


def _face_connected_components(mesh: trimesh.Trimesh) -> List[np.ndarray]:
    face_count = int(len(mesh.faces))
    adjacency = [[] for _ in range(face_count)]
    for face_a, face_b in np.asarray(mesh.face_adjacency, dtype=np.int64):
        adjacency[int(face_a)].append(int(face_b))
        adjacency[int(face_b)].append(int(face_a))

    visited = np.zeros(face_count, dtype=bool)
    components: List[np.ndarray] = []
    for start_face in range(face_count):
        if visited[start_face]:
            continue

        stack = [start_face]
        visited[start_face] = True
        component = []
        while stack:
            face_idx = stack.pop()
            component.append(face_idx)
            for neighbor_face in adjacency[face_idx]:
                if not visited[neighbor_face]:
                    visited[neighbor_face] = True
                    stack.append(neighbor_face)

        components.append(np.asarray(component, dtype=np.int64))

    return components


def _remove_tiny_face_components(
    mesh: trimesh.Trimesh, min_face_count: int = 128
) -> trimesh.Trimesh:
    components = _face_connected_components(mesh)
    if len(components) <= 1:
        return mesh

    largest_size = max(len(component) for component in components)
    kept_components = [
        component
        for component in components
        if len(component) == largest_size or len(component) >= min_face_count
    ]
    kept_face_count = sum(len(component) for component in kept_components)
    if kept_face_count == len(mesh.faces):
        return mesh

    kept_faces = np.concatenate(kept_components, axis=0)
    kept_faces.sort()
    filtered_mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int64)[kept_faces],
        process=False,
    )
    filtered_mesh.fix_normals()
    return filtered_mesh


def _extract_boundary_loops(mesh: trimesh.Trimesh) -> List[np.ndarray]:
    edges = np.sort(mesh.edges_sorted.reshape(-1, 2), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if boundary_edges.shape[0] == 0:
        return []

    adjacency: Dict[int, List[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))

    loops: List[np.ndarray] = []
    visited_vertices = set()
    for start in adjacency:
        if start in visited_vertices:
            continue

        component = []
        stack = [start]
        visited_vertices.add(start)
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for neighbor in adjacency[vertex]:
                if neighbor not in visited_vertices:
                    visited_vertices.add(neighbor)
                    stack.append(neighbor)

        if any(len(adjacency[v]) != 2 for v in component):
            continue

        start_vertex = min(component)
        ordered = [start_vertex]
        previous = None
        current = start_vertex

        while True:
            neighbors = adjacency[current]
            if previous is None:
                next_vertex = min(neighbors)
            else:
                next_candidates = [v for v in neighbors if v != previous]
                if not next_candidates:
                    break
                next_vertex = next_candidates[0]

            if next_vertex == start_vertex:
                break

            ordered.append(next_vertex)
            previous, current = current, next_vertex

            if len(ordered) > len(component) + 1:
                break

        if len(ordered) >= 3:
            loops.append(np.asarray(ordered, dtype=np.int64))

    return loops


def _triangulate_planar_loop(
    vertices: np.ndarray, loop_vertex_indices: np.ndarray, planarity_tol: float
) -> np.ndarray:
    loop_vertices = vertices[loop_vertex_indices]
    centroid = loop_vertices.mean(axis=0)
    centered = loop_vertices - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2]
    normal = vh[2]
    plane_distance = centered @ normal
    if float(np.abs(plane_distance).max()) > planarity_tol:
        return np.zeros((0, 3), dtype=np.int64)

    projected = centered @ basis.T
    if _polygon_signed_area(projected) < 0.0:
        loop_vertex_indices = loop_vertex_indices[::-1]
        projected = projected[::-1]

    active = list(range(len(loop_vertex_indices)))
    triangles: List[List[int]] = []
    eps = 1e-12

    while len(active) > 3:
        ear_found = False
        for idx in range(len(active)):
            prev_local = active[(idx - 1) % len(active)]
            curr_local = active[idx]
            next_local = active[(idx + 1) % len(active)]

            a = projected[prev_local]
            b = projected[curr_local]
            c = projected[next_local]
            cross = float(np.cross(b - a, c - b))
            if cross <= eps:
                continue

            is_ear = True
            for other_local in active:
                if other_local in (prev_local, curr_local, next_local):
                    continue
                if _point_in_triangle_2d(projected[other_local], a, b, c, eps=eps):
                    is_ear = False
                    break

            if not is_ear:
                continue

            triangles.append(
                [
                    int(loop_vertex_indices[prev_local]),
                    int(loop_vertex_indices[curr_local]),
                    int(loop_vertex_indices[next_local]),
                ]
            )
            del active[idx]
            ear_found = True
            break

        if not ear_found:
            root = active[0]
            for idx in range(1, len(active) - 1):
                triangles.append(
                    [
                        int(loop_vertex_indices[root]),
                        int(loop_vertex_indices[active[idx]]),
                        int(loop_vertex_indices[active[idx + 1]]),
                    ]
                )
            active = []

    if len(active) == 3:
        triangles.append(
            [
                int(loop_vertex_indices[active[0]]),
                int(loop_vertex_indices[active[1]]),
                int(loop_vertex_indices[active[2]]),
            ]
        )

    return np.asarray(triangles, dtype=np.int64)


def _cap_planar_boundary_loops(
    mesh: trimesh.Trimesh, planarity_tol: float = 1e-3
) -> Tuple[trimesh.Trimesh, int]:
    original_face_count = int(len(mesh.faces))
    loops = _extract_boundary_loops(mesh)
    if not loops:
        return mesh, original_face_count

    cap_faces = []
    for loop_vertex_indices in loops:
        triangles = _triangulate_planar_loop(
            np.asarray(mesh.vertices, dtype=np.float64),
            loop_vertex_indices,
            planarity_tol=planarity_tol,
        )
        if triangles.shape[0] > 0:
            cap_faces.append(triangles)

    if not cap_faces:
        return mesh, original_face_count

    capped_mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.concatenate([np.asarray(mesh.faces, dtype=np.int64), *cap_faces], axis=0),
        process=False,
    )
    capped_mesh.fix_normals()
    return capped_mesh, original_face_count


def _sanitize_array(array: np.ndarray, nan: float = 0.0) -> np.ndarray:
    return np.nan_to_num(array, nan=nan, posinf=nan, neginf=-nan if nan != 0.0 else 0.0)


def _stable_seed(seed: int, text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (seed + int(digest[:16], 16)) % (2**32 - 1)


def _assign_face_labels(faces: np.ndarray, vertex_labels: np.ndarray) -> np.ndarray:
    face_vertex_labels = vertex_labels[faces]
    face_labels = np.zeros(len(faces), dtype=np.int32)

    for face_idx, labels in enumerate(face_vertex_labels):
        positive_labels = labels[labels > 0]
        if positive_labels.size == 0:
            continue
        unique_labels, counts = np.unique(positive_labels, return_counts=True)
        order = np.lexsort((unique_labels, -counts))
        face_labels[face_idx] = int(unique_labels[order[0]])

    return face_labels


def _sample_points_from_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    face_indices: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if count <= 0 or face_indices.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    areas = face_areas[face_indices].astype(np.float64)
    positive = areas > 0
    if np.any(positive):
        face_indices = face_indices[positive]
        areas = areas[positive]
        probabilities = areas / areas.sum()
    else:
        probabilities = None

    chosen_faces = rng.choice(face_indices, size=count, replace=True, p=probabilities)
    triangles = vertices[faces[chosen_faces]]

    r1 = rng.random(count)
    r2 = rng.random(count)
    sqrt_r1 = np.sqrt(r1)

    points = (
        (1.0 - sqrt_r1)[:, None] * triangles[:, 0]
        + (sqrt_r1 * (1.0 - r2))[:, None] * triangles[:, 1]
        + (sqrt_r1 * r2)[:, None] * triangles[:, 2]
    )
    normals = face_normals[chosen_faces]

    return points.astype(np.float32), normals.astype(np.float32)


def _voxelize_surface_points(
    points: np.ndarray,
    normals: np.ndarray,
    resolution: int,
    bbox_min: float = -1.0,
    bbox_max: float = 1.0,
) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    voxel_size = (bbox_max - bbox_min) / float(resolution)
    grid_indices = np.floor((points - bbox_min) / voxel_size).astype(np.int64)
    grid_indices = np.clip(grid_indices, 0, resolution - 1)

    linear_indices = (
        grid_indices[:, 0]
        + resolution * (grid_indices[:, 1] + resolution * grid_indices[:, 2])
    )
    unique_linear, first_occurrence, inverse = np.unique(
        linear_indices, return_inverse=True, return_index=True
    )
    unique_indices = grid_indices[first_occurrence]

    centers = bbox_min + (unique_indices.astype(np.float32) + 0.5) * voxel_size
    accumulated_normals = np.zeros((unique_linear.shape[0], 3), dtype=np.float64)
    np.add.at(accumulated_normals, inverse, normals.astype(np.float64))
    normal_norms = np.linalg.norm(accumulated_normals, axis=1, keepdims=True)
    averaged_normals = accumulated_normals / np.clip(normal_norms, 1e-12, None)

    surface = np.concatenate([centers, averaged_normals.astype(np.float32)], axis=1)
    return _sanitize_array(surface.astype(np.float32))


def _voxelized_surface_to_mesh(
    surface: np.ndarray,
    resolution: int,
    bbox_min: float = -1.0,
    bbox_max: float = 1.0,
) -> trimesh.Trimesh:
    centers = np.asarray(surface[:, :3], dtype=np.float64)
    voxel_pitch = float(bbox_max - bbox_min) / float(resolution)
    return trimesh.voxel.ops.multibox(
        centers,
        pitch=voxel_pitch,
        remove_internal_faces=True,
    )


def _save_voxelized_variant_meshes(
    coarse_variants: List[np.ndarray],
    sharp_variants: List[np.ndarray],
    coarse_resolutions: np.ndarray,
    tooth_resolutions: np.ndarray,
    output_root: Path,
    file_ext: str,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for variant_idx, (coarse_surface, sharp_surface) in enumerate(zip(coarse_variants, sharp_variants)):
        coarse_mesh = _voxelized_surface_to_mesh(
            coarse_surface,
            resolution=int(coarse_resolutions[variant_idx]),
        )
        sharp_resolution = int(np.max(tooth_resolutions[variant_idx]))
        sharp_mesh = _voxelized_surface_to_mesh(
            sharp_surface,
            resolution=sharp_resolution,
        )

        coarse_path = output_root / f"coarse_variant_{variant_idx:02d}.{file_ext}"
        sharp_path = output_root / f"sharp_variant_{variant_idx:02d}.{file_ext}"
        coarse_mesh.export(coarse_path)
        sharp_mesh.export(sharp_path)


def _fps_or_resample(
    samples: np.ndarray, target_count: int, rng: np.random.Generator
) -> np.ndarray:
    if samples.shape[0] == 0:
        return np.zeros((target_count, 6), dtype=np.float32)
    if samples.shape[0] == target_count:
        return samples.astype(np.float32)
    if samples.shape[0] < target_count:
        indices = rng.choice(samples.shape[0], size=target_count, replace=True)
        return samples[indices].astype(np.float32)

    if fpsample is not None:
        try:
            indices = fpsample.bucket_fps_kdline_sampling(samples[:, :3], target_count, h=5)
            return samples[np.asarray(indices, dtype=np.int64)].astype(np.float32)
        except Exception:
            pass

    indices = rng.choice(samples.shape[0], size=target_count, replace=False)
    return samples[indices].astype(np.float32)


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    face_normals: np.ndarray
    face_areas: np.ndarray
    all_face_indices: np.ndarray
    original_face_count: int
    tooth_labels: np.ndarray
    tooth_face_indices: Dict[int, np.ndarray]
    tooth_face_pool: np.ndarray
    sdf_fn: SDF


def _load_mesh_data(obj_path: Path, json_path: Path) -> MeshData:
    mesh = _normalize_mesh(_load_mesh(obj_path))
    mesh = _remove_tiny_face_components(mesh)

    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "labels" not in payload:
        raise ValueError(f"Missing 'labels' in {json_path}")

    vertex_labels = np.asarray(payload["labels"], dtype=np.int32)
    if vertex_labels.ndim != 1:
        raise ValueError(f"Expected 1D 'labels' array in {json_path}")
    if vertex_labels.shape[0] != len(mesh.vertices):
        raise ValueError(
            f"Vertex/label mismatch for {obj_path}: {len(mesh.vertices)} vertices vs "
            f"{vertex_labels.shape[0]} labels"
        )
    mesh, original_face_count = _cap_planar_boundary_loops(mesh)
    if not mesh.is_watertight:
        raise ValueError(f"Mesh is not watertight: {obj_path}")

    original_faces = np.asarray(mesh.faces[:original_face_count], dtype=np.int64)
    face_labels = _assign_face_labels(original_faces, vertex_labels)
    tooth_labels = np.array(sorted(int(label) for label in np.unique(face_labels) if label > 0), dtype=np.int32)
    tooth_face_indices = {
        int(label): np.where(face_labels == label)[0].astype(np.int64)
        for label in tooth_labels
        if np.any(face_labels == label)
    }
    if not tooth_face_indices:
        raise ValueError(f"No tooth faces found after labeling: {obj_path}")

    tooth_face_pool = np.concatenate(list(tooth_face_indices.values())).astype(np.int64)
    return MeshData(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        face_normals=np.asarray(mesh.face_normals, dtype=np.float32),
        face_areas=np.asarray(mesh.area_faces, dtype=np.float32),
        all_face_indices=np.arange(len(mesh.faces), dtype=np.int64),
        original_face_count=original_face_count,
        tooth_labels=tooth_labels,
        tooth_face_indices=tooth_face_indices,
        tooth_face_pool=tooth_face_pool,
        sdf_fn=SDF(mesh.vertices, mesh.faces),
    )


def _build_coarse_variant(
    mesh_data: MeshData,
    resolution: int,
    input_points: int,
    surface_pool_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    points, normals = _sample_points_from_faces(
        mesh_data.vertices,
        mesh_data.faces,
        mesh_data.face_normals,
        mesh_data.face_areas,
        mesh_data.all_face_indices,
        surface_pool_size,
        rng,
    )
    voxelized = _voxelize_surface_points(points, normals, resolution)
    return _fps_or_resample(voxelized, input_points, rng)


def _build_sharp_variant(
    mesh_data: MeshData,
    tooth_resolutions: np.ndarray,
    input_points: int,
    min_surface_samples: int,
    max_surface_samples: int,
    surface_sample_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    parts = []
    for tooth_label, tooth_resolution in zip(mesh_data.tooth_labels.tolist(), tooth_resolutions.tolist()):
        face_indices = mesh_data.tooth_face_indices[int(tooth_label)]
        sample_count = int(max(min_surface_samples, tooth_resolution * tooth_resolution * surface_sample_scale))
        sample_count = min(sample_count, max_surface_samples)
        points, normals = _sample_points_from_faces(
            mesh_data.vertices,
            mesh_data.faces,
            mesh_data.face_normals,
            mesh_data.face_areas,
            face_indices,
            sample_count,
            rng,
        )
        voxelized = _voxelize_surface_points(points, normals, int(tooth_resolution))
        if voxelized.shape[0] > 0:
            parts.append(voxelized)

    if not parts:
        raise RuntimeError("Sharp/tooth voxelization produced no samples")

    merged = np.concatenate(parts, axis=0)
    return _fps_or_resample(merged, input_points, rng)


def _build_sharp_near_surface(
    mesh_data: MeshData,
    surface_samples: int,
    noise_scales: List[float],
    rng: np.random.Generator,
) -> np.ndarray:
    tooth_points, _ = _sample_points_from_faces(
        mesh_data.vertices,
        mesh_data.faces,
        mesh_data.face_normals,
        mesh_data.face_areas,
        mesh_data.tooth_face_pool,
        surface_samples,
        rng,
    )
    near_surface_points = []
    for scale in noise_scales:
        noise = rng.normal(scale=scale, size=tooth_points.shape).astype(np.float32)
        near_surface_points.append((tooth_points + noise).astype(np.float32))
    query_points = np.concatenate(near_surface_points, axis=0)
    sdf = np.asarray(mesh_data.sdf_fn(query_points), dtype=np.float32).reshape(-1, 1)
    return _sanitize_array(np.concatenate([query_points, sdf], axis=1))


def _build_rand_points(
    mesh_data: MeshData,
    surface_samples: int,
    space_samples: int,
    noise_scales: List[float],
    sdf_bbox_min: float,
    sdf_bbox_max: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    surface_points, _ = _sample_points_from_faces(
        mesh_data.vertices,
        mesh_data.faces,
        mesh_data.face_normals,
        mesh_data.face_areas,
        mesh_data.all_face_indices,
        surface_samples,
        rng,
    )
    near_surface_points = []
    for scale in noise_scales:
        noise = rng.normal(scale=scale, size=surface_points.shape).astype(np.float32)
        near_surface_points.append((surface_points + noise).astype(np.float32))
    near_surface_points = np.concatenate(near_surface_points, axis=0)

    space_points = rng.uniform(
        low=sdf_bbox_min, high=sdf_bbox_max, size=(space_samples, 3)
    ).astype(np.float32)
    query_points = np.concatenate([near_surface_points, space_points], axis=0)
    sdf = np.asarray(mesh_data.sdf_fn(query_points), dtype=np.float32).reshape(-1, 1)
    return _sanitize_array(np.concatenate([query_points, sdf], axis=1)), int(near_surface_points.shape[0])


def _process_pair(
    obj_path: Path,
    json_path: Path,
    data_root: Path,
    samples_root: Path,
    args: argparse.Namespace,
) -> Path:
    mesh_data = _load_mesh_data(obj_path, json_path)
    relative_obj_path = obj_path.relative_to(data_root)

    scan_seed = _stable_seed(args.seed, relative_obj_path.as_posix())
    seed_rng = np.random.default_rng(scan_seed)
    variant_seeds = seed_rng.integers(0, 2**32 - 1, size=args.n_variants, dtype=np.uint32)

    coarse_variants = []
    sharp_variants = []
    coarse_resolutions = np.zeros(args.n_variants, dtype=np.int32)
    tooth_resolutions = np.zeros((args.n_variants, mesh_data.tooth_labels.shape[0]), dtype=np.int32)

    for variant_idx, variant_seed in enumerate(variant_seeds.tolist()):
        rng = np.random.default_rng(int(variant_seed))
        coarse_resolution = int(rng.choice(args.coarse_resolutions))
        per_tooth_resolutions = rng.choice(
            args.tooth_resolutions,
            size=mesh_data.tooth_labels.shape[0],
            replace=True,
        ).astype(np.int32)

        coarse_surface = _build_coarse_variant(
            mesh_data=mesh_data,
            resolution=coarse_resolution,
            input_points=args.input_points,
            surface_pool_size=args.coarse_surface_pool,
            rng=rng,
        )
        sharp_surface = _build_sharp_variant(
            mesh_data=mesh_data,
            tooth_resolutions=per_tooth_resolutions,
            input_points=args.input_points,
            min_surface_samples=args.tooth_surface_min_samples,
            max_surface_samples=args.tooth_surface_max_samples,
            surface_sample_scale=args.tooth_surface_sample_scale,
            rng=rng,
        )

        coarse_variants.append(coarse_surface)
        sharp_variants.append(sharp_surface)
        coarse_resolutions[variant_idx] = coarse_resolution
        tooth_resolutions[variant_idx] = per_tooth_resolutions

    sharp_rng = np.random.default_rng(scan_seed ^ 0xA5A5A5A5)
    sharp_near_surface = _build_sharp_near_surface(
        mesh_data=mesh_data,
        surface_samples=args.sharp_surface_samples,
        noise_scales=args.sharp_noise_scales,
        rng=sharp_rng,
    )

    rand_rng = np.random.default_rng(scan_seed ^ 0x5A5A5A5A)
    rand_points, rand_points_near_count = _build_rand_points(
        mesh_data=mesh_data,
        surface_samples=args.rand_surface_samples,
        space_samples=args.space_point_samples,
        noise_scales=args.rand_noise_scales,
        sdf_bbox_min=args.sdf_bbox_min,
        sdf_bbox_max=args.sdf_bbox_max,
        rng=rand_rng,
    )

    output_path = samples_root / relative_obj_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        fps_coarse_surface=np.stack(coarse_variants, axis=1).astype(np.float32),
        fps_sharp_surface=np.stack(sharp_variants, axis=1).astype(np.float32),
        sharp_near_surface=sharp_near_surface.astype(np.float32),
        rand_points=rand_points.astype(np.float32),
        coarse_resolutions=coarse_resolutions.astype(np.int32),
        tooth_resolutions=tooth_resolutions.astype(np.int32),
        variant_seeds=variant_seeds.astype(np.uint32),
        present_tooth_labels=mesh_data.tooth_labels.astype(np.int32),
        rand_points_near_count=np.array(rand_points_near_count, dtype=np.int32),
    )

    if args.save_voxelized_meshes:
        voxel_mesh_root = args.voxelized_meshes_root / relative_obj_path.with_suffix("")
        _save_voxelized_variant_meshes(
            coarse_variants=coarse_variants,
            sharp_variants=sharp_variants,
            coarse_resolutions=coarse_resolutions,
            tooth_resolutions=tooth_resolutions,
            output_root=voxel_mesh_root,
            file_ext=args.voxelized_mesh_format,
        )

    return output_path.resolve()


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_splits(processed_files: List[str], output_root: Path, seed: int, train_ratio: float, val_ratio: float) -> None:
    files = list(processed_files)
    random.Random(seed).shuffle(files)

    if not files:
        raise RuntimeError("No processed files available to split")

    total = len(files)
    n_train = int(train_ratio * total)
    n_val = int(val_ratio * total)
    n_train = max(1, min(n_train, total))
    n_val = max(0, min(n_val, total - n_train))

    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]

    if not test:
        if val:
            test = val[-1:]
            val = val[:-1]
        else:
            test = train[-1:]
            train = train[:-1] or test

    _write_json(output_root / "train.json", train)
    _write_json(output_root / "val.json", val)
    _write_json(output_root / "test.json", test)


def _discover_pairs(data_root: Path) -> Tuple[List[Tuple[Path, Path]], List[dict]]:
    pairs = []
    invalid_entries = []

    for obj_path in sorted(data_root.rglob("*.obj")):
        json_path = obj_path.with_suffix(".json")
        if not json_path.exists():
            invalid_entries.append(
                {
                    "mesh_path": str(obj_path.resolve()),
                    "json_path": str(json_path.resolve()),
                    "reason": "Missing matching JSON file",
                }
            )
            continue
        pairs.append((obj_path, json_path))

    return pairs, invalid_entries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=_default_data_root())
    parser.add_argument("--output_root", type=str, default=_default_output_root())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.90)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--n_variants", type=int, default=5)
    parser.add_argument("--input_points", type=int, default=32768)
    parser.add_argument("--coarse_resolutions", type=_parse_int_list, default=DEFAULT_COARSE_RESOLUTIONS)
    parser.add_argument("--tooth_resolutions", type=_parse_int_list, default=DEFAULT_TOOTH_RESOLUTIONS)
    parser.add_argument("--coarse_surface_pool", type=int, default=200000)
    parser.add_argument("--tooth_surface_min_samples", type=int, default=8000)
    parser.add_argument("--tooth_surface_max_samples", type=int, default=50000)
    parser.add_argument("--tooth_surface_sample_scale", type=float, default=2.5)
    parser.add_argument("--sharp_surface_samples", type=int, default=50000)
    parser.add_argument("--rand_surface_samples", type=int, default=200000)
    parser.add_argument("--space_point_samples", type=int, default=200000)
    parser.add_argument("--sharp_noise_scales", type=_parse_float_list, default=DEFAULT_SHARP_NOISE_SCALES)
    parser.add_argument("--rand_noise_scales", type=_parse_float_list, default=DEFAULT_RAND_NOISE_SCALES)
    parser.add_argument("--sdf_bbox_min", type=float, default=-1.05)
    parser.add_argument("--sdf_bbox_max", type=float, default=1.05)
    parser.add_argument("--save_voxelized_meshes", action="store_true")
    parser.add_argument("--voxelized_mesh_format", type=str, default="obj", choices=["obj", "ply", "stl"])
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.data_root:
        raise RuntimeError("Missing data_root. Set $SCRATCH or pass --data_root explicitly.")
    if not args.output_root:
        raise RuntimeError("Missing output_root. Set $SCRATCH or pass --output_root explicitly.")

    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    samples_root = output_root / "samples"
    args.voxelized_meshes_root = output_root / "voxelized_meshes"

    if not data_root.is_dir():
        raise RuntimeError(f"Data root does not exist: {data_root}")
    if args.train_ratio <= 0 or args.train_ratio >= 1:
        raise RuntimeError("train_ratio must be in (0, 1)")
    if args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise RuntimeError("train_ratio + val_ratio must be < 1")

    output_root.mkdir(parents=True, exist_ok=True)
    samples_root.mkdir(parents=True, exist_ok=True)

    pairs, invalid_entries = _discover_pairs(data_root)
    processed_files: List[str] = []

    for obj_path, json_path in tqdm(pairs, desc="Preprocessing voxel-smoothing pairs"):
        try:
            output_path = _process_pair(obj_path, json_path, data_root, samples_root, args)
            processed_files.append(str(output_path))
        except Exception as exc:
            invalid_entries.append(
                {
                    "mesh_path": str(obj_path.resolve()),
                    "json_path": str(json_path.resolve()),
                    "reason": str(exc),
                }
            )

    processed_files = sorted(processed_files)
    _write_json(output_root / "processed_meshes.json", processed_files)
    _write_json(output_root / "invalid_meshes.json", invalid_entries)

    if processed_files:
        _write_splits(
            processed_files=processed_files,
            output_root=output_root,
            seed=args.split_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Processed meshes: {len(processed_files)}")
    print(f"Invalid meshes: {len(invalid_entries)}")
    if processed_files:
        print(f"Split files: {output_root / 'train.json'} {output_root / 'val.json'} {output_root / 'test.json'}")


if __name__ == "__main__":
    main()
