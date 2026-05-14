import os
from dataclasses import dataclass

import numpy as np


@dataclass
class FusionStats:
    input_points: int = 0
    valid_points: int = 0
    created: int = 0
    fused: int = 0
    conflicts: int = 0
    skipped_low_conf: int = 0

    def to_dict(self):
        return {
            "input_points": int(self.input_points),
            "valid_points": int(self.valid_points),
            "created": int(self.created),
            "fused": int(self.fused),
            "conflicts": int(self.conflicts),
            "skipped_low_conf": int(self.skipped_low_conf),
        }


class VoxelSurfelFusion:
    """Small confidence-aware surfel map backed by a voxel hash.

    This is intentionally conservative: repeat observations close to an existing
    surfel are fused into that surfel; observations near but inconsistent with an
    existing surfel are counted as conflicts and skipped instead of becoming a
    second ghost surface.
    """

    def __init__(
        self,
        voxel_size=0.02,
        fusion_radius=None,
        conflict_radius=None,
        min_weight=1e-4,
        max_weight=100.0,
        min_export_weight=0.0,
        min_export_observations=1,
        max_export_points=0,
        seed=42,
    ):
        self.voxel_size = float(voxel_size)
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        self.fusion_radius = float(fusion_radius or self.voxel_size * 1.5)
        self.conflict_radius = float(conflict_radius or self.voxel_size * 3.0)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.min_export_weight = float(min_export_weight)
        self.min_export_observations = int(min_export_observations)
        self.max_export_points = int(max_export_points)
        self.rng = np.random.default_rng(int(seed))
        self._voxels = {}
        self._surfels = []
        self.total_integrations = 0
        self.total_observations = 0
        self.total_conflicts = 0

    @classmethod
    def from_config(cls, config, seed=42):
        cfg = config or {}
        return cls(
            voxel_size=cfg.get("voxel_size", 0.02),
            fusion_radius=cfg.get("fusion_radius"),
            conflict_radius=cfg.get("conflict_radius"),
            min_weight=cfg.get("min_weight", 1e-4),
            max_weight=cfg.get("max_weight", 100.0),
            min_export_weight=cfg.get("min_export_weight", 0.0),
            min_export_observations=cfg.get("min_export_observations", 1),
            max_export_points=cfg.get("max_export_points", 0),
            seed=cfg.get("seed", seed),
        )

    def reset(self):
        self._voxels = {}
        self._surfels = []
        self.total_integrations = 0
        self.total_observations = 0
        self.total_conflicts = 0

    @property
    def surfel_count(self):
        return len(self._surfels)

    def _voxel_key(self, point):
        return tuple(np.floor(point / self.voxel_size).astype(np.int64).tolist())

    def _neighbor_keys(self, key):
        x, y, z = key
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    yield (x + dx, y + dy, z + dz)

    def _find_nearest(self, point):
        key = self._voxel_key(point)
        best_idx = None
        best_dist = np.inf
        for neighbor_key in self._neighbor_keys(key):
            for surfel_idx in self._voxels.get(neighbor_key, ()):
                surfel = self._surfels[surfel_idx]
                dist = float(np.linalg.norm(point - surfel["position"]))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = surfel_idx
        return best_idx, best_dist

    def _add_surfel(self, point, color, weight):
        key = self._voxel_key(point)
        surfel_idx = len(self._surfels)
        self._surfels.append(
            {
                "position": point.astype(np.float32, copy=True),
                "color": color.astype(np.float32, copy=True),
                "weight": float(min(weight, self.max_weight)),
                "observations": 1,
                "variance": 0.0,
                "conflicts": 0,
                "voxel_key": key,
            }
        )
        self._voxels.setdefault(key, []).append(surfel_idx)

    def _reindex_surfel_if_needed(self, surfel_idx, old_key, new_key):
        if old_key == new_key:
            return
        old_bucket = self._voxels.get(old_key)
        if old_bucket is not None:
            try:
                old_bucket.remove(surfel_idx)
            except ValueError:
                pass
            if not old_bucket:
                self._voxels.pop(old_key, None)
        self._voxels.setdefault(new_key, []).append(surfel_idx)

    def _update_surfel(self, surfel_idx, point, color, weight):
        surfel = self._surfels[surfel_idx]
        old_position = surfel["position"]
        old_color = surfel["color"]
        old_weight = float(surfel["weight"])
        old_key = surfel["voxel_key"]
        obs_weight = max(float(weight), self.min_weight)
        total_weight = max(old_weight + obs_weight, self.min_weight)

        new_position = (old_position * old_weight + point * obs_weight) / total_weight
        new_color = (old_color * old_weight + color * obs_weight) / total_weight

        residual = float(np.linalg.norm(point - new_position))
        old_variance = float(surfel["variance"])
        new_variance = (old_variance * old_weight + residual * residual * obs_weight) / total_weight

        surfel["position"] = new_position.astype(np.float32, copy=False)
        surfel["color"] = np.clip(new_color, 0.0, 255.0).astype(np.float32, copy=False)
        surfel["weight"] = float(min(total_weight, self.max_weight))
        surfel["observations"] = int(surfel["observations"]) + 1
        surfel["variance"] = float(new_variance)
        new_key = self._voxel_key(surfel["position"])
        surfel["voxel_key"] = new_key
        self._reindex_surfel_if_needed(surfel_idx, old_key, new_key)

    def integrate(
        self,
        points,
        colors,
        confs,
        conf_threshold=0.0,
        sample_ratio=1.0,
        depth_range=None,
    ):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(colors).reshape(-1, 3).astype(np.float32)
        confs = np.asarray(confs, dtype=np.float32).reshape(-1)
        stats = FusionStats(input_points=len(points))

        finite_mask = np.isfinite(points).all(axis=1) & np.isfinite(confs)
        conf_mask = confs >= float(conf_threshold)
        valid_mask = finite_mask & conf_mask
        if depth_range is not None:
            low, high = depth_range
            if low is not None or high is not None:
                z = points[:, 2]
                if low is not None:
                    valid_mask &= z >= float(low)
                if high is not None:
                    valid_mask &= z <= float(high)

        stats.skipped_low_conf = int(np.count_nonzero(finite_mask & ~conf_mask))
        indices = np.flatnonzero(valid_mask)
        if 0.0 < sample_ratio < 1.0 and len(indices) > 0:
            keep_count = max(1, int(len(indices) * float(sample_ratio)))
            indices = self.rng.choice(indices, size=keep_count, replace=False)

        stats.valid_points = int(len(indices))
        for idx in indices:
            point = points[idx]
            color = colors[idx]
            weight = max(float(confs[idx]), self.min_weight)

            nearest_idx, nearest_dist = self._find_nearest(point)
            if nearest_idx is not None and nearest_dist <= self.fusion_radius:
                self._update_surfel(nearest_idx, point, color, weight)
                stats.fused += 1
            elif nearest_idx is not None and nearest_dist <= self.conflict_radius:
                self._surfels[nearest_idx]["conflicts"] = (
                    int(self._surfels[nearest_idx]["conflicts"]) + 1
                )
                stats.conflicts += 1
            else:
                self._add_surfel(point, color, weight)
                stats.created += 1

        self.total_integrations += 1
        self.total_observations += int(stats.valid_points)
        self.total_conflicts += int(stats.conflicts)
        return stats.to_dict()

    def export_arrays(self):
        positions = []
        colors = []
        weights = []
        observations = []
        variances = []
        for surfel in self._surfels:
            if float(surfel["weight"]) < self.min_export_weight:
                continue
            if int(surfel["observations"]) < self.min_export_observations:
                continue
            positions.append(surfel["position"])
            colors.append(surfel["color"])
            weights.append(float(surfel["weight"]))
            observations.append(int(surfel["observations"]))
            variances.append(float(surfel["variance"]))

        if not positions:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float32),
            )

        positions = np.asarray(positions, dtype=np.float32)
        colors = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 255.0).astype(np.uint8)
        weights = np.asarray(weights, dtype=np.float32)
        observations = np.asarray(observations, dtype=np.int32)
        variances = np.asarray(variances, dtype=np.float32)

        if self.max_export_points > 0 and len(positions) > self.max_export_points:
            indices = self.rng.choice(len(positions), size=self.max_export_points, replace=False)
            positions = positions[indices]
            colors = colors[indices]
            weights = weights[indices]
            observations = observations[indices]
            variances = variances[indices]

        return positions, colors, weights, observations, variances

    def export_ply(self, output_path):
        positions, colors, _, _, _ = self.export_arrays()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as handle:
            write_ply(handle, positions, colors)
        return {
            "surfel_count": int(self.surfel_count),
            "exported_point_count": int(len(positions)),
            "total_integrations": int(self.total_integrations),
            "total_observations": int(self.total_observations),
            "total_conflicts": int(self.total_conflicts),
            "voxel_size": float(self.voxel_size),
            "fusion_radius": float(self.fusion_radius),
            "conflict_radius": float(self.conflict_radius),
        }


def write_ply(handle, points, colors):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    handle.write(("\n".join(header) + "\n").encode("ascii"))
    if len(points) == 0:
        return
    structured = np.zeros(
        len(points),
        dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("red", np.uint8),
            ("green", np.uint8),
            ("blue", np.uint8),
        ],
    )
    structured["x"] = points[:, 0]
    structured["y"] = points[:, 1]
    structured["z"] = points[:, 2]
    structured["red"] = colors[:, 0]
    structured["green"] = colors[:, 1]
    structured["blue"] = colors[:, 2]
    handle.write(structured.tobytes())
