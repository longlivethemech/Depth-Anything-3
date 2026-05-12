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
    low_conf_integrated: int = 0

    def to_dict(self):
        return {
            "input_points": int(self.input_points),
            "valid_points": int(self.valid_points),
            "created": int(self.created),
            "fused": int(self.fused),
            "conflicts": int(self.conflicts),
            "skipped_low_conf": int(self.skipped_low_conf),
            "low_conf_integrated": int(self.low_conf_integrated),
        }


class VoxelSurfelFusion:
    """Confidence-aware surfel map backed by a voxel hash.

    Low/medium-confidence observations are integrated as candidate surfels
    instead of being dropped immediately. A surfel becomes confirmed when it has
    enough weight from one strong observation or enough repeated support from
    consistent observations. Only confirmed surfels are exported.
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
        confirm_observations=2,
        confirm_weight=1.5,
        instant_confirm_weight=3.0,
        instant_confirm_low_conf=False,
        candidate_conflict_policy="skip_against_confirmed",
        max_integrate_points=0,
        seed=42,
    ):
        self.voxel_size = float(voxel_size)
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        self.fusion_radius = float(fusion_radius or self.voxel_size * 1.5)
        self.conflict_radius = float(conflict_radius or self.voxel_size * 3.0)
        self._neighbor_span = max(1, int(np.ceil(self.conflict_radius / self.voxel_size)))
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.min_export_weight = float(min_export_weight)
        self.min_export_observations = int(min_export_observations)
        self.max_export_points = int(max_export_points)
        self.confirm_observations = max(int(confirm_observations), 1)
        self.confirm_weight = float(confirm_weight)
        self.instant_confirm_weight = float(instant_confirm_weight)
        self.instant_confirm_low_conf = bool(instant_confirm_low_conf)
        self.candidate_conflict_policy = str(candidate_conflict_policy or "skip_against_confirmed")
        self.max_integrate_points = max(int(max_integrate_points or 0), 0)
        self.rng = np.random.default_rng(int(seed))
        self._voxels = {}
        self._surfels = []
        self.total_integrations = 0
        self.total_observations = 0
        self.total_conflicts = 0
        self.total_low_conf_integrated = 0

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
            confirm_observations=cfg.get("confirm_observations", 2),
            confirm_weight=cfg.get("confirm_weight", 1.5),
            instant_confirm_weight=cfg.get("instant_confirm_weight", 3.0),
            instant_confirm_low_conf=cfg.get("instant_confirm_low_conf", False),
            candidate_conflict_policy=cfg.get(
                "candidate_conflict_policy",
                "skip_against_confirmed",
            ),
            max_integrate_points=cfg.get("max_integrate_points", 0),
            seed=cfg.get("seed", seed),
        )

    def reset(self):
        self._voxels = {}
        self._surfels = []
        self.total_integrations = 0
        self.total_observations = 0
        self.total_conflicts = 0
        self.total_low_conf_integrated = 0

    @property
    def surfel_count(self):
        return len(self._surfels)

    def _voxel_key(self, point):
        return tuple(np.floor(point / self.voxel_size).astype(np.int64).tolist())

    def _neighbor_keys(self, key):
        x, y, z = key
        span = self._neighbor_span
        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                for dz in range(-span, span + 1):
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

    def _maybe_confirm_surfel(self, surfel):
        if surfel["state"] == "confirmed":
            return False
        weight = float(surfel["weight"])
        support_observations = int(surfel["support_observations"])
        has_high_conf = int(surfel["high_conf_observations"]) > 0
        if weight >= self.instant_confirm_weight and (
            has_high_conf or self.instant_confirm_low_conf
        ):
            surfel["state"] = "confirmed"
            surfel["confirmed_reason"] = "instant_weight"
            return True
        if support_observations >= self.confirm_observations and weight >= self.confirm_weight:
            surfel["state"] = "confirmed"
            surfel["confirmed_reason"] = "repeated_weight"
            return True
        return False

    def _add_surfel(self, point, color, weight, low_conf, observation_id):
        key = self._voxel_key(point)
        surfel_idx = len(self._surfels)
        integration_id = int(self.total_integrations)
        support_ids = {int(observation_id)}
        surfel = {
            "position": point.astype(np.float32, copy=True),
            "color": color.astype(np.float32, copy=True),
            "weight": float(min(weight, self.max_weight)),
            "observations": 1,
            "support_observations": 1,
            "_support_ids": support_ids,
            "variance": 0.0,
            "conflicts": 0,
            "voxel_key": key,
            "state": "candidate",
            "first_seen_integration": integration_id,
            "last_seen_integration": integration_id,
            "low_conf_observations": int(bool(low_conf)),
            "high_conf_observations": int(not low_conf),
            "confirmed_reason": None,
        }
        self._maybe_confirm_surfel(surfel)
        self._surfels.append(surfel)
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

    def _update_surfel(self, surfel_idx, point, color, weight, low_conf, observation_id):
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
        support_ids = surfel.setdefault("_support_ids", set())
        observation_id = int(observation_id)
        if observation_id not in support_ids:
            support_ids.add(observation_id)
            surfel["support_observations"] = int(surfel["support_observations"]) + 1
        surfel["variance"] = float(new_variance)
        surfel["last_seen_integration"] = int(self.total_integrations)
        surfel["low_conf_observations"] = int(surfel["low_conf_observations"]) + int(bool(low_conf))
        surfel["high_conf_observations"] = int(surfel["high_conf_observations"]) + int(not low_conf)
        self._maybe_confirm_surfel(surfel)
        new_key = self._voxel_key(surfel["position"])
        surfel["voxel_key"] = new_key
        self._reindex_surfel_if_needed(surfel_idx, old_key, new_key)

    def _map_state_counts(self):
        candidate_count = 0
        confirmed_count = 0
        low_conf_confirmed = 0
        for surfel in self._surfels:
            if surfel["state"] == "confirmed":
                confirmed_count += 1
                if int(surfel["low_conf_observations"]) > 0:
                    low_conf_confirmed += 1
            else:
                candidate_count += 1
        return candidate_count, confirmed_count, low_conf_confirmed

    def integrate(
        self,
        points,
        colors,
        confs,
        conf_threshold=0.0,
        sample_ratio=1.0,
        depth_range=None,
        low_conf_threshold=None,
        observation_ids=None,
    ):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(colors).reshape(-1, 3).astype(np.float32)
        confs = np.asarray(confs, dtype=np.float32).reshape(-1)
        if observation_ids is None:
            observation_ids = np.full(len(points), int(self.total_integrations), dtype=np.int64)
        else:
            observation_ids = np.asarray(observation_ids, dtype=np.int64).reshape(-1)
            if len(observation_ids) != len(points):
                raise ValueError("observation_ids must match points length")
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

        low_conf_threshold = float(low_conf_threshold) if low_conf_threshold is not None else float(conf_threshold)
        stats.skipped_low_conf = int(np.count_nonzero(finite_mask & ~conf_mask))
        indices = np.flatnonzero(valid_mask)
        if len(indices) > 0:
            keep_count = len(indices)
            if 0.0 < sample_ratio < 1.0:
                keep_count = max(1, int(len(indices) * float(sample_ratio)))
            if self.max_integrate_points > 0:
                keep_count = min(keep_count, self.max_integrate_points)
            if keep_count < len(indices):
                indices = self.rng.choice(indices, size=keep_count, replace=False)

        stats.valid_points = int(len(indices))
        for idx in indices:
            point = points[idx]
            color = colors[idx]
            weight = max(float(confs[idx]), self.min_weight)
            low_conf = bool(float(confs[idx]) < low_conf_threshold)
            observation_id = int(observation_ids[idx])
            stats.low_conf_integrated += int(low_conf)

            nearest_idx, nearest_dist = self._find_nearest(point)
            if nearest_idx is not None and nearest_dist <= self.fusion_radius:
                self._update_surfel(nearest_idx, point, color, weight, low_conf, observation_id)
                stats.fused += 1
            elif nearest_idx is not None and nearest_dist <= self.conflict_radius:
                nearest = self._surfels[nearest_idx]
                if self.candidate_conflict_policy in ("create_candidate", "create_against_candidate"):
                    self._add_surfel(point, color, weight, low_conf, observation_id)
                    stats.created += 1
                elif (
                    self.candidate_conflict_policy == "skip_against_confirmed"
                    and nearest["state"] != "confirmed"
                ):
                    self._add_surfel(point, color, weight, low_conf, observation_id)
                    stats.created += 1
                else:
                    nearest["conflicts"] = int(nearest["conflicts"]) + 1
                    stats.conflicts += 1
            else:
                self._add_surfel(point, color, weight, low_conf, observation_id)
                stats.created += 1

        self.total_integrations += 1
        self.total_observations += int(stats.valid_points)
        self.total_conflicts += int(stats.conflicts)
        self.total_low_conf_integrated += int(stats.low_conf_integrated)
        result = stats.to_dict()
        candidate_count, confirmed_count, low_conf_confirmed = self._map_state_counts()
        result.update({
            "candidate_count": int(candidate_count),
            "confirmed_count": int(confirmed_count),
            "low_conf_confirmed": int(low_conf_confirmed),
            "conflict_count": int(self.total_conflicts),
            "support_observation_count": int(sum(int(s["support_observations"]) for s in self._surfels)),
            "max_integrate_points": int(self.max_integrate_points),
        })
        return result

    def export_arrays(self):
        positions = []
        colors = []
        weights = []
        observations = []
        support_observations = []
        variances = []
        for surfel in self._surfels:
            if surfel["state"] != "confirmed":
                continue
            if float(surfel["weight"]) < self.min_export_weight:
                continue
            if int(surfel["observations"]) < self.min_export_observations:
                continue
            positions.append(surfel["position"])
            colors.append(surfel["color"])
            weights.append(float(surfel["weight"]))
            observations.append(int(surfel["observations"]))
            support_observations.append(int(surfel["support_observations"]))
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
            support_observations = np.asarray(support_observations, dtype=np.int32)[indices]
            variances = variances[indices]

        return positions, colors, weights, observations, variances

    def export_ply(self, output_path):
        positions, colors, _, _, _ = self.export_arrays()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as handle:
            write_ply(handle, positions, colors)
        candidate_count, confirmed_count, low_conf_confirmed = self._map_state_counts()
        return {
            "surfel_count": int(self.surfel_count),
            "candidate_count": int(candidate_count),
            "confirmed_count": int(confirmed_count),
            "low_conf_integrated": int(self.total_low_conf_integrated),
            "low_conf_confirmed": int(low_conf_confirmed),
            "conflict_count": int(self.total_conflicts),
            "exported_point_count": int(len(positions)),
            "total_integrations": int(self.total_integrations),
            "total_observations": int(self.total_observations),
            "total_conflicts": int(self.total_conflicts),
            "voxel_size": float(self.voxel_size),
            "fusion_radius": float(self.fusion_radius),
            "conflict_radius": float(self.conflict_radius),
            "confirm_observations": int(self.confirm_observations),
            "confirm_weight": float(self.confirm_weight),
            "instant_confirm_weight": float(self.instant_confirm_weight),
            "instant_confirm_low_conf": bool(self.instant_confirm_low_conf),
            "candidate_conflict_policy": self.candidate_conflict_policy,
            "max_integrate_points": int(self.max_integrate_points),
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
