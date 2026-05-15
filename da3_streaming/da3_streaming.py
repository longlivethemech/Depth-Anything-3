# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long)

import argparse
import gc
import glob
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from loop_utils.alignment_torch import (
    apply_sim3_direct_torch,
    depth_to_point_cloud_optimized_torch,
)
from loop_utils.config_utils import load_config
from loop_utils.loop_detector import LoopDetector
from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import (
    accumulate_sim3_transforms,
    compute_sim3_ab,
    merge_ply_files,
    precompute_scale_chunks_with_depth,
    process_loop_list,
    save_confident_pointcloud_batch,
    warmup_numba,
    weighted_align_point_maps,
)
from safetensors.torch import load_file

from depth_anything_3.api import DepthAnything3

matplotlib.use("Agg")


def timing_now():
    return time.perf_counter()


def print_timing(label, started_at):
    elapsed = time.perf_counter() - started_at
    print(f"[TIMING] {label}: {elapsed:.3f}s")
    return elapsed


def effective_cpu_count(default=4):
    cpu_max_path = "/sys/fs/cgroup/cpu.max"
    try:
        if os.path.exists(cpu_max_path):
            quota_str, period_str = open(cpu_max_path, "r", encoding="utf-8").read().strip().split()
            if quota_str != "max":
                quota = int(quota_str)
                period = int(period_str)
                if quota > 0 and period > 0:
                    return max(1, (quota + period - 1) // period)
    except Exception:
        pass

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        pass

    return max(1, os.cpu_count() or default)


def resolve_device_dtype(device=None, dtype=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        if str(device).startswith("cuda") and torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return device, dtype


def _abs_path(path, base_dir=None):
    if path is None:
        return None
    path = os.fspath(path)
    if os.path.isabs(path):
        return path
    if base_dir is None:
        base_dir = os.getcwd()
    return os.path.abspath(os.path.join(os.fspath(base_dir), path))


def _path_marker(path):
    if path is None:
        return None
    abs_path = os.path.abspath(os.fspath(path))
    marker = {"path": abs_path, "exists": os.path.exists(abs_path)}
    if marker["exists"]:
        stat = os.stat(abs_path)
        marker.update({"mtime_ns": stat.st_mtime_ns, "size": stat.st_size})
    return marker


def _config_with_absolute_weight_paths(config, base_dir=None):
    resolved = deepcopy(config)
    weights = resolved.get("Weights", {})
    for key in ("DA3_CONFIG", "DA3", "SALAD"):
        if key in weights and weights[key]:
            weights[key] = _abs_path(weights[key], base_dir)
    return resolved


def load_job_config(config_path, base_dir=None, absolute_paths=False):
    resolved_config_path = _abs_path(config_path, base_dir) if absolute_paths else config_path
    config = load_config(resolved_config_path)
    if absolute_paths:
        config = _config_with_absolute_weight_paths(config, base_dir)
    return config, resolved_config_path


@dataclass
class DA3RuntimeAssets:
    model: object
    device: str
    dtype: object
    loop_detector_model: object = None
    loop_detector_device: object = None
    cache_key: str = ""
    numba_warmed: bool = False


def runtime_assets_cache_key(config, config_path=None, device=None, dtype=None):
    device, dtype = resolve_device_dtype(device=device, dtype=dtype)
    weights = config.get("Weights", {})
    key_payload = {
        "da3_config": _path_marker(weights.get("DA3_CONFIG")),
        "da3_weights": _path_marker(weights.get("DA3")),
        "device": str(device),
        "dtype": str(dtype),
        "loop_enable": bool(config.get("Model", {}).get("loop_enable")),
    }
    if key_payload["loop_enable"]:
        key_payload["salad_weights"] = _path_marker(weights.get("SALAD"))
    return json.dumps(key_payload, sort_keys=True)


def build_runtime_assets(config, config_path=None, device=None, dtype=None):
    assets_started = timing_now()
    device, dtype = resolve_device_dtype(device=device, dtype=dtype)

    print("Preloading DA3 runtime assets...")
    da3_config_started = timing_now()
    with open(config["Weights"]["DA3_CONFIG"]) as f:
        da3_config = json.load(f)
    print_timing("da3.runtime_assets.load_da3_json_config", da3_config_started)

    construct_model_started = timing_now()
    model = DepthAnything3(**da3_config)
    print_timing("da3.runtime_assets.construct_depthanything3", construct_model_started)

    load_weights_started = timing_now()
    weight = load_file(config["Weights"]["DA3"])
    print_timing("da3.runtime_assets.load_safetensors", load_weights_started)

    load_state_dict_started = timing_now()
    model.load_state_dict(weight, strict=False)
    print_timing("da3.runtime_assets.load_state_dict", load_state_dict_started)

    model_to_device_started = timing_now()
    model.eval()
    model = model.to(device)
    print_timing("da3.runtime_assets.model_eval_to_device", model_to_device_started)

    loop_detector_model = None
    loop_detector_device = None
    if config["Model"]["loop_enable"]:
        loop_detector_construct_started = timing_now()
        loop_detector = LoopDetector(image_dir="", output="", config=config)
        print_timing(
            "da3.runtime_assets.loop_detector_construct", loop_detector_construct_started
        )
        loop_detector_load_started = timing_now()
        loop_detector_model, loop_detector_device = loop_detector.load_model()
        print_timing(
            "da3.runtime_assets.loop_detector_load_model", loop_detector_load_started
        )
        loop_detector.image_paths = None
        loop_detector.descriptors = None
        loop_detector.loop_closures = None
        del loop_detector

    assets = DA3RuntimeAssets(
        model=model,
        device=device,
        dtype=dtype,
        loop_detector_model=loop_detector_model,
        loop_detector_device=loop_detector_device,
        cache_key=runtime_assets_cache_key(config, config_path, device=device, dtype=dtype),
    )
    print_timing("da3.runtime_assets.total", assets_started)
    return assets


def depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics, device=None):
    """
    depth: [N, H, W] numpy array or torch tensor
    intrinsics: [N, 3, 3] numpy array or torch tensor
    extrinsics: [N, 3, 4] (w2c) numpy array or torch tensor
    Returns: point_cloud_world: [N, H, W, 3] same type as input
    """
    input_is_numpy = False
    if isinstance(depth, np.ndarray):
        input_is_numpy = True

        depth_tensor = torch.tensor(depth, dtype=torch.float32)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
        extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float32)

        if device is not None:
            depth_tensor = depth_tensor.to(device)
            intrinsics_tensor = intrinsics_tensor.to(device)
            extrinsics_tensor = extrinsics_tensor.to(device)
    else:
        depth_tensor = depth
        intrinsics_tensor = intrinsics
        extrinsics_tensor = extrinsics

    if device is not None:
        depth_tensor = depth_tensor.to(device)
        intrinsics_tensor = intrinsics_tensor.to(device)
        extrinsics_tensor = extrinsics_tensor.to(device)

    # main logic

    N, H, W = depth_tensor.shape

    device = depth_tensor.device

    u = torch.arange(W, device=device).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v = torch.arange(H, device=device).float().view(1, H, 1, 1).expand(N, H, W, 1)
    ones = torch.ones((N, H, W, 1), device=device)
    pixel_coords = torch.cat([u, v, ones], dim=-1)

    intrinsics_inv = torch.inverse(intrinsics_tensor)  # [N, 3, 3]
    camera_coords = torch.einsum("nij,nhwj->nhwi", intrinsics_inv, pixel_coords)
    camera_coords = camera_coords * depth_tensor.unsqueeze(-1)
    camera_coords_homo = torch.cat([camera_coords, ones], dim=-1)

    extrinsics_4x4 = torch.zeros(N, 4, 4, device=device)
    extrinsics_4x4[:, :3, :4] = extrinsics_tensor
    extrinsics_4x4[:, 3, 3] = 1.0

    c2w = torch.inverse(extrinsics_4x4)
    world_coords_homo = torch.einsum("nij,nhwj->nhwi", c2w, camera_coords_homo)
    point_cloud_world = world_coords_homo[..., :3]

    if input_is_numpy:
        point_cloud_world = point_cloud_world.cpu().numpy()

    return point_cloud_world


def remove_duplicates(data_list):
    """
    data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {}
    result = []

    for item in data_list:
        if item[0] == item[2]:
            continue

        key = (item[0], item[2])

        if key not in seen.keys():
            seen[key] = True
            result.append(item)

    return result


class DA3_Streaming:
    def __init__(self, image_dir, save_dir, config, runtime_assets=None):
        init_started = timing_now()
        self.config = config

        self.chunk_size = self.config["Model"]["chunk_size"]
        self.overlap = self.config["Model"]["overlap"]
        self.overlap_s = 0
        self.overlap_e = self.overlap - self.overlap_s
        self.conf_threshold = 1.5
        self.seed = 42
        self.runtime_assets = runtime_assets
        self._uses_preloaded_model = runtime_assets is not None
        self._uses_preloaded_loop_detector = (
            runtime_assets is not None and runtime_assets.loop_detector_model is not None
        )
        if runtime_assets is not None:
            self.device = runtime_assets.device
            self.dtype = runtime_assets.dtype
        else:
            self.device, self.dtype = resolve_device_dtype()

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir

        self.result_unaligned_dir = os.path.join(save_dir, "_tmp_results_unaligned")
        self.result_aligned_dir = os.path.join(save_dir, "_tmp_results_aligned")
        self.result_loop_dir = os.path.join(save_dir, "_tmp_results_loop")
        self.result_output_dir = os.path.join(save_dir, "results_output")
        self.pcd_dir = os.path.join(save_dir, "pcd")
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)

        self.all_camera_poses = []
        self.all_camera_intrinsics = []

        self.delete_temp_files = self.config["Model"]["delete_temp_files"]

        print("Loading model...")

        if runtime_assets is not None:
            self.model = runtime_assets.model
            print("[RESIDENT] Reusing preloaded DA3 model")
        else:
            da3_config_started = timing_now()
            with open(self.config["Weights"]["DA3_CONFIG"]) as f:
                config = json.load(f)
            print_timing("da3.init.load_da3_json_config", da3_config_started)

            construct_model_started = timing_now()
            self.model = DepthAnything3(**config)
            print_timing("da3.init.construct_depthanything3", construct_model_started)

            load_weights_started = timing_now()
            weight = load_file(self.config["Weights"]["DA3"])
            print_timing("da3.init.load_safetensors", load_weights_started)

            load_state_dict_started = timing_now()
            self.model.load_state_dict(weight, strict=False)
            print_timing("da3.init.load_state_dict", load_state_dict_started)

            model_to_device_started = timing_now()
            self.model.eval()
            self.model = self.model.to(self.device)
            print_timing("da3.init.model_eval_to_device", model_to_device_started)

        self.skyseg_session = None

        self.chunk_indices = None  # [(begin_idx, end_idx), ...]

        self.loop_list = []  # e.g. [(1584, 139), ...]

        self.loop_optimizer = Sim3LoopOptimizer(self.config)

        self.sim3_list = []  # [(s [1,], R [3,3], T [3,]), ...]

        self.loop_sim3_list = []  # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]

        self.loop_predict_list = []
        self.streaming_loop_detection_history = []
        self.streaming_loop_detection_seen = set()
        self.streaming_loop_correction_seen_chunk_pairs = set()
        self.streaming_loop_correction_pending_windows = []
        self.streaming_map_epoch = 0
        self.streaming_frame_rotation_cache = {}
        self.streaming_chunk_rotation_cache = {}

        self.loop_enable = self.config["Model"]["loop_enable"]

        if self.loop_enable:
            loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
            loop_detector_construct_started = timing_now()
            self.loop_detector = LoopDetector(
                image_dir=image_dir, output=loop_info_save_path, config=self.config
            )
            print_timing("da3.init.loop_detector_construct", loop_detector_construct_started)
            if self._uses_preloaded_loop_detector:
                self.loop_detector.model = runtime_assets.loop_detector_model
                self.loop_detector.device = runtime_assets.loop_detector_device
                print("[RESIDENT] Reusing preloaded loop detector model")
            else:
                loop_detector_load_started = timing_now()
                self.loop_detector.load_model()
                print_timing("da3.init.loop_detector_load_model", loop_detector_load_started)

        print("init done.")
        print_timing("da3.init.total", init_started)

    def get_loop_pairs(self):
        loop_pairs_started = timing_now()
        self.loop_detector.run()
        print_timing("da3.loop.loop_detector_run", loop_pairs_started)
        get_loop_list_started = timing_now()
        loop_list = self.loop_detector.get_loop_list()
        print_timing("da3.loop.get_loop_list", get_loop_list_started)
        print_timing("da3.loop.get_loop_pairs_total", loop_pairs_started)
        return loop_list

    def get_streaming_loop_closures(self):
        loop_pairs_started = timing_now()
        self.loop_detector.run(save_results=False, incremental_only=True)
        print_timing("da3.streaming_loop.loop_detector_run_incremental", loop_pairs_started)
        closures = list(getattr(self.loop_detector, "loop_closures", None) or [])
        print_timing("da3.streaming_loop.get_loop_closures_total", loop_pairs_started)
        return closures

    def _rotation_angle_degrees(self, rotation):
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape != (3, 3):
            return 0.0
        cos_theta = (float(np.trace(rotation)) - 1.0) / 2.0
        cos_theta = min(1.0, max(-1.0, cos_theta))
        return float(np.degrees(np.arccos(cos_theta)))

    def _rotation_vector_degrees(self, rotation):
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape != (3, 3):
            return np.zeros(3, dtype=np.float64)
        cos_theta = (float(np.trace(rotation)) - 1.0) / 2.0
        cos_theta = min(1.0, max(-1.0, cos_theta))
        angle = float(np.arccos(cos_theta))
        if angle < 1e-8:
            return np.zeros(3, dtype=np.float64)
        skew = np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=np.float64,
        )
        sin_theta = float(np.sin(angle))
        if abs(sin_theta) < 1e-6:
            axis = skew
            norm = float(np.linalg.norm(axis))
            if norm < 1e-8:
                return np.zeros(3, dtype=np.float64)
            axis = axis / norm
        else:
            axis = skew / (2.0 * sin_theta)
        return axis * float(np.degrees(angle))

    def _empty_rotation_metrics(self, source):
        return {
            "source": source,
            "total_deg": 0.0,
            "peak_deg": 0.0,
            "net_deg": 0.0,
            "unwrapped_net_deg": 0.0,
            "principal_net_deg": 0.0,
            "directionality": 0.0,
            "measured_steps": 0,
        }

    def _chunk_path_rotation_metrics(self, chunk_idx_a, chunk_idx_b):
        if not self.sim3_list:
            return self._empty_rotation_metrics("chunk_sim3_path")

        start = min(int(chunk_idx_a), int(chunk_idx_b)) + 1
        end = max(int(chunk_idx_a), int(chunk_idx_b)) + 1
        total_degrees = 0.0
        peak_degrees = 0.0
        cumulative_vector = np.zeros(3, dtype=np.float64)
        net_rotation = np.eye(3)
        measured_steps = 0
        for chunk_idx in range(start, end):
            transform_idx = chunk_idx - 1
            if transform_idx < 0 or transform_idx >= len(self.sim3_list):
                continue
            _, rotation, _ = self.sim3_list[transform_idx]
            step_vector = self._rotation_vector_degrees(rotation)
            total_degrees += float(np.linalg.norm(step_vector))
            cumulative_vector += step_vector
            peak_degrees = max(peak_degrees, float(np.linalg.norm(cumulative_vector)))
            net_rotation = rotation @ net_rotation
            measured_steps += 1
        return {
            "source": "chunk_sim3_path",
            "total_deg": float(total_degrees),
            "peak_deg": float(peak_degrees),
            # net_deg is intentionally the unwrapped path displacement, not the
            # principal SO(3) angle, so a full turn remains near 360 deg.
            "net_deg": float(np.linalg.norm(cumulative_vector)),
            "unwrapped_net_deg": float(np.linalg.norm(cumulative_vector)),
            "principal_net_deg": self._rotation_angle_degrees(net_rotation),
            "directionality": float(np.linalg.norm(cumulative_vector) / max(total_degrees, 1e-9)),
            "measured_steps": measured_steps,
        }

    def _loop_rotation_class(self, path_rotation_deg):
        if path_rotation_deg >= 270.0:
            return "full_turn_revisit", 3
        if path_rotation_deg >= 150.0:
            return "wide_turn_revisit", 2
        if path_rotation_deg >= 45.0:
            return "turn_revisit", 1
        return "low_rotation_revisit", 0

    def _frame_chunk_index(self, frame_idx):
        frame_idx = int(frame_idx)
        if not self.chunk_indices:
            return None
        for chunk_idx in range(len(self.chunk_indices) - 1, -1, -1):
            begin, end = self.chunk_indices[chunk_idx]
            if int(begin) <= frame_idx < int(end):
                return chunk_idx
        return None

    def _streaming_chunk_global_camera_rotations(self, chunk_idx):
        chunk_idx = int(chunk_idx)
        cached = self.streaming_chunk_rotation_cache.get(chunk_idx)
        if cached is not None:
            return cached
        if chunk_idx < 0 or chunk_idx >= len(self.chunk_indices):
            return None

        chunk_path = os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy")
        if not os.path.exists(chunk_path):
            return None
        chunk_data = np.load(chunk_path, allow_pickle=True).item()
        rotations = []
        sim3 = None
        if chunk_idx > 0:
            accumulated_sim3 = accumulate_sim3_transforms(list(self.sim3_list))
            if chunk_idx - 1 < len(accumulated_sim3):
                sim3 = accumulated_sim3[chunk_idx - 1]
        for local_idx in range(len(chunk_data.extrinsics)):
            w2c = np.eye(4)
            w2c[:3, :] = chunk_data.extrinsics[local_idx]
            c2w = np.linalg.inv(w2c)
            rotation = c2w[:3, :3]
            if sim3 is not None:
                _, sim3_rotation, _ = sim3
                rotation = sim3_rotation @ rotation
            rotations.append(rotation)
        self.streaming_chunk_rotation_cache[chunk_idx] = rotations
        return rotations

    def _streaming_frame_global_camera_rotation(self, frame_idx):
        frame_idx = int(frame_idx)
        if frame_idx in self.streaming_frame_rotation_cache:
            return self.streaming_frame_rotation_cache[frame_idx]
        chunk_idx = self._frame_chunk_index(frame_idx)
        if chunk_idx is None:
            return None
        begin, _ = self.chunk_indices[chunk_idx]
        local_idx = frame_idx - int(begin)
        rotations = self._streaming_chunk_global_camera_rotations(chunk_idx)
        if rotations is None or local_idx < 0 or local_idx >= len(rotations):
            return None
        rotation = rotations[local_idx]
        self.streaming_frame_rotation_cache[frame_idx] = rotation
        return rotation

    def _camera_path_rotation_metrics(self, frame_idx_a, frame_idx_b):
        start = min(int(frame_idx_a), int(frame_idx_b))
        end = max(int(frame_idx_a), int(frame_idx_b))
        if end <= start:
            return None

        total_degrees = 0.0
        peak_degrees = 0.0
        cumulative_vector = np.zeros(3, dtype=np.float64)
        previous_rotation = None
        first_rotation = None
        last_rotation = None
        measured_steps = 0
        for frame_idx in range(start, end + 1):
            rotation = self._streaming_frame_global_camera_rotation(frame_idx)
            if rotation is None:
                continue
            if first_rotation is None:
                first_rotation = rotation
            last_rotation = rotation
            if previous_rotation is not None:
                relative_rotation = rotation @ previous_rotation.T
                step_vector = self._rotation_vector_degrees(relative_rotation)
                total_degrees += float(np.linalg.norm(step_vector))
                cumulative_vector += step_vector
                peak_degrees = max(peak_degrees, float(np.linalg.norm(cumulative_vector)))
                measured_steps += 1
            previous_rotation = rotation
        if measured_steps == 0:
            return None
        net_degrees = 0.0
        if first_rotation is not None and last_rotation is not None:
            net_degrees = self._rotation_angle_degrees(last_rotation @ first_rotation.T)
        return {
            "source": "camera_pose_path",
            "total_deg": float(total_degrees),
            "peak_deg": float(peak_degrees),
            # net_deg is the unwrapped accumulated camera-path rotation.  Keep
            # principal_net_deg separately because its SO(3) angle is folded to
            # 0..180 deg and cannot distinguish full turns from returning.
            "net_deg": float(np.linalg.norm(cumulative_vector)),
            "unwrapped_net_deg": float(np.linalg.norm(cumulative_vector)),
            "principal_net_deg": float(net_degrees),
            "directionality": float(np.linalg.norm(cumulative_vector) / max(total_degrees, 1e-9)),
            "measured_steps": int(measured_steps),
        }

    def _loop_path_rotation_metrics(self, center_a, center_b, chunk_a, chunk_b):
        camera_path_rotation = self._camera_path_rotation_metrics(center_a, center_b)
        if camera_path_rotation is not None:
            return camera_path_rotation
        return self._chunk_path_rotation_metrics(chunk_a, chunk_b)

    def _clamp01(self, value):
        return float(max(0.0, min(1.0, float(value))))

    def _linear_score(self, value, low, high):
        if high <= low:
            return 0.0
        return self._clamp01((float(value) - float(low)) / (float(high) - float(low)))

    def _loop_similarity_matrix(self):
        similarity_matrix = getattr(self.loop_detector, "similarity_matrix", None)
        if similarity_matrix is not None:
            return similarity_matrix
        descriptors = getattr(self.loop_detector, "descriptors", None)
        if descriptors is None:
            return None
        descriptors = descriptors.numpy()
        similarity_matrix = descriptors @ descriptors.T
        self.loop_detector.similarity_matrix = similarity_matrix
        return similarity_matrix

    def _window_support_score(self, center_a, center_b, half_window=8):
        similarity_matrix = self._loop_similarity_matrix()
        if similarity_matrix is None:
            return 0.0
        frame_count = int(similarity_matrix.shape[0])
        a0 = max(0, int(center_a) - half_window)
        a1 = min(frame_count, int(center_a) + half_window + 1)
        b0 = max(0, int(center_b) - half_window)
        b1 = min(frame_count, int(center_b) + half_window + 1)
        if a0 >= a1 or b0 >= b1:
            return 0.0
        local = similarity_matrix[a0:a1, b0:b1]
        if local.size == 0:
            return 0.0
        return self._linear_score(float(np.percentile(local, 90)), 0.60, 0.95)

    def _loop_priority_components(
        self,
        similarity,
        window_support_score,
        frame_gap,
        chunk_gap,
        rotation_metrics,
    ):
        total_deg = float(rotation_metrics.get("total_deg", 0.0))
        peak_deg = float(rotation_metrics.get("peak_deg", 0.0))
        unwrapped_net_deg = float(
            rotation_metrics.get("unwrapped_net_deg", rotation_metrics.get("net_deg", 0.0))
        )
        one_way_ratio = self._clamp01(unwrapped_net_deg / max(total_deg, 1e-9))

        sim_score = self._linear_score(similarity, 0.60, 0.95)
        evidence = 0.85 * sim_score + 0.15 * self._clamp01(window_support_score)

        full_turn_score = 1.0 if (
            peak_deg >= 270.0
            and unwrapped_net_deg >= 270.0
            and one_way_ratio >= 0.70
        ) else 0.0
        rot_score = self._clamp01(peak_deg / 360.0)
        frame_gap_score = self._clamp01(float(frame_gap) / 240.0)
        chunk_gap_score = self._clamp01(float(chunk_gap) / 8.0)
        opportunity = (
            0.40 * rot_score
            + 0.25 * frame_gap_score
            + 0.15 * chunk_gap_score
            + 0.15 * one_way_ratio
            + 0.05 * full_turn_score
        )
        opportunity_boost = 0.30 + 0.70 * opportunity
        priority_score = evidence * opportunity_boost
        return {
            "sim_score": float(sim_score),
            "window_support_score": float(window_support_score),
            "evidence_score": float(evidence),
            "rot_score": float(rot_score),
            "frame_gap_score": float(frame_gap_score),
            "chunk_gap_score": float(chunk_gap_score),
            "one_way_score": float(one_way_ratio),
            "full_turn_score": float(full_turn_score),
            "opportunity_score": float(opportunity),
            "opportunity_boost": float(opportunity_boost),
            "priority_score": float(priority_score),
        }

    def _loop_pair_score_map(self):
        closures = getattr(self.loop_detector, "loop_closures", None) or []
        scores = {}
        for idx_a, idx_b, similarity in closures:
            key = tuple(sorted((int(idx_a), int(idx_b))))
            scores[key] = max(float(similarity), scores.get(key, float("-inf")))
        return scores

    def _best_loop_pair_for_ranges(self, range_a, range_b, loop_pair_scores):
        best_pair = None
        best_similarity = 0.0
        a0, a1 = int(range_a[0]), int(range_a[1])
        b0, b1 = int(range_b[0]), int(range_b[1])
        for pair_key, similarity in loop_pair_scores.items():
            idx_a, idx_b = int(pair_key[0]), int(pair_key[1])
            matches_direct = a0 <= idx_a <= a1 and b0 <= idx_b <= b1
            matches_swapped = a0 <= idx_b <= a1 and b0 <= idx_a <= b1
            if not matches_direct and not matches_swapped:
                continue
            if best_pair is None or float(similarity) > best_similarity:
                best_similarity = float(similarity)
                best_pair = [idx_a, idx_b]
        return best_pair, best_similarity

    def _annotate_streaming_loop_window(self, item, loop_pair_scores, source_loop_pair=None):
        chunk_a = int(item[0])
        range_a = (int(item[1][0]), int(item[1][1]))
        chunk_b = int(item[2])
        range_b = (int(item[3][0]), int(item[3][1]))
        center_a = (range_a[0] + range_a[1]) // 2
        center_b = (range_b[0] + range_b[1]) // 2
        frame_gap = abs(center_a - center_b)
        chunk_gap = abs(chunk_a - chunk_b)
        if source_loop_pair is not None and len(source_loop_pair) >= 2:
            source_pair = [int(source_loop_pair[0]), int(source_loop_pair[1])]
            if len(source_loop_pair) >= 3:
                similarity = float(source_loop_pair[2])
            else:
                similarity = float(
                    loop_pair_scores.get(tuple(sorted(source_pair)), 0.0)
                )
        else:
            source_pair, similarity = self._best_loop_pair_for_ranges(
                range_a,
                range_b,
                loop_pair_scores,
            )
        rotation_metrics = self._loop_path_rotation_metrics(
            center_a,
            center_b,
            chunk_a,
            chunk_b,
        )
        path_rotation_deg = float(rotation_metrics.get("peak_deg", 0.0))
        rotation_class, rotation_class_rank = self._loop_rotation_class(path_rotation_deg)
        window_support_score = self._window_support_score(center_a, center_b)
        priority_components = self._loop_priority_components(
            similarity,
            window_support_score,
            frame_gap,
            chunk_gap,
            rotation_metrics,
        )
        return {
            "item": item,
            "chunk_a": chunk_a,
            "range_a": range_a,
            "chunk_b": chunk_b,
            "range_b": range_b,
            "center_a": center_a,
            "center_b": center_b,
            "chunk_gap": chunk_gap,
            "frame_gap": frame_gap,
            "similarity": similarity,
            "source_pair": source_pair,
            "path_rotation_deg": path_rotation_deg,
            "path_rotation_total_deg": float(rotation_metrics.get("total_deg", 0.0)),
            "path_rotation_peak_deg": float(rotation_metrics.get("peak_deg", 0.0)),
            "path_rotation_net_deg": float(rotation_metrics.get("net_deg", 0.0)),
            "path_rotation_unwrapped_net_deg": float(rotation_metrics.get("unwrapped_net_deg", rotation_metrics.get("net_deg", 0.0))),
            "path_rotation_principal_net_deg": float(rotation_metrics.get("principal_net_deg", rotation_metrics.get("net_deg", 0.0))),
            "path_rotation_directionality": float(rotation_metrics.get("directionality", 0.0)),
            "path_rotation_measured_steps": int(rotation_metrics.get("measured_steps", 0)),
            "path_rotation_source": rotation_metrics.get("source", "unknown"),
            "rotation_class": rotation_class,
            "rotation_class_rank": rotation_class_rank,
            **priority_components,
        }

    def _same_loop_cluster(self, candidate, kept, config):
        frame_radius = int(config["frame_radius"])
        if (
            abs(int(candidate["center_a"]) - int(kept["center_a"])) > frame_radius
            or abs(int(candidate["center_b"]) - int(kept["center_b"])) > frame_radius
        ):
            return False
        if str(candidate.get("rotation_class")) != str(kept.get("rotation_class")):
            return False
        rotation_checks = (
            ("path_rotation_peak_deg", "peak_deg_radius"),
            ("path_rotation_unwrapped_net_deg", "net_deg_radius"),
            ("path_rotation_total_deg", "total_deg_radius"),
        )
        for value_key, radius_key in rotation_checks:
            if abs(float(candidate.get(value_key, 0.0)) - float(kept.get(value_key, 0.0))) > float(config[radius_key]):
                return False
        return True

    def _cluster_ranked_loop_windows(self, ranked_loop_results):
        salad_cfg = self.config["Loop"]["SALAD"]
        config = {
            "enabled": bool(salad_cfg.get("cluster_nms_enabled", True)),
            "frame_radius": int(salad_cfg.get("cluster_frame_radius", 10)),
            "peak_deg_radius": float(salad_cfg.get("cluster_peak_deg_radius", 30.0)),
            "net_deg_radius": float(salad_cfg.get("cluster_net_deg_radius", 30.0)),
            "total_deg_radius": float(salad_cfg.get("cluster_total_deg_radius", 60.0)),
        }
        if not config["enabled"]:
            for item in ranked_loop_results:
                item["cluster_size"] = int(item.get("cluster_size", 1))
            return ranked_loop_results, {
                **config,
                "input_count": len(ranked_loop_results),
                "kept_count": len(ranked_loop_results),
                "suppressed_count": 0,
            }

        kept = []
        for candidate in ranked_loop_results:
            matched = None
            for existing in kept:
                if self._same_loop_cluster(candidate, existing, config):
                    matched = existing
                    break
            if matched is not None:
                matched["cluster_size"] = int(matched.get("cluster_size", 1)) + 1
                continue
            candidate["cluster_size"] = 1
            kept.append(candidate)
        return kept, {
            **config,
            "input_count": len(ranked_loop_results),
            "kept_count": len(kept),
            "suppressed_count": len(ranked_loop_results) - len(kept),
        }

    def detect_streaming_loop_candidates(
        self,
        chunk_idx,
        min_chunk_gap=0,
        min_frame_gap=0,
        max_new_windows=0,
    ):
        loop_detection_started = timing_now()
        if not self.loop_enable:
            return {
                "enabled": False,
                "chunk_idx": int(chunk_idx),
                "reason": "loop disabled in config",
                "loop_pairs": [],
                "loop_windows": [],
                "new_loop_windows": [],
            }
        if not hasattr(self, "loop_detector"):
            loop_info_save_path = os.path.join(self.output_dir, "loop_closures.txt")
            loop_detector_construct_started = timing_now()
            self.loop_detector = LoopDetector(
                image_dir=self.img_dir, output=loop_info_save_path, config=self.config
            )
            print_timing("da3.streaming_loop.loop_detector_construct", loop_detector_construct_started)
            if self._uses_preloaded_loop_detector and self.runtime_assets is not None:
                self.loop_detector.model = self.runtime_assets.loop_detector_model
                self.loop_detector.device = self.runtime_assets.loop_detector_device
                print("[RESIDENT] Reusing preloaded loop detector model for streaming loop detection")
            else:
                loop_detector_load_started = timing_now()
                self.loop_detector.load_model()
                print_timing("da3.streaming_loop.loop_detector_load_model", loop_detector_load_started)

        self.loop_detector.image_paths = None
        self.loop_detector.loop_closures = None
        self.streaming_frame_rotation_cache = {}
        self.streaming_chunk_rotation_cache = {}

        self.loop_list = self.get_streaming_loop_closures()
        loop_pair_scores = self._loop_pair_score_map()
        process_loop_list_started = timing_now()
        loop_pairs_for_windows = [(int(a), int(b)) for a, b, _ in self.loop_list]
        loop_results = process_loop_list(
            self.chunk_indices,
            loop_pairs_for_windows,
            half_window=int(self.config["Model"]["loop_chunk_size"] / 2),
        )
        annotated_by_key = {}
        for item, loop_pair in zip(loop_results, self.loop_list):
            annotated = self._annotate_streaming_loop_window(
                item,
                loop_pair_scores,
                source_loop_pair=loop_pair,
            )
            key = (
                int(annotated["chunk_a"]),
                int(annotated["range_a"][0]),
                int(annotated["range_a"][1]),
                int(annotated["chunk_b"]),
                int(annotated["range_b"][0]),
                int(annotated["range_b"][1]),
            )
            existing = annotated_by_key.get(key)
            if existing is None or annotated["priority_score"] > existing["priority_score"]:
                annotated_by_key[key] = annotated
        annotated_loop_results = sorted(
            annotated_by_key.values(),
            key=lambda x: (
                x["priority_score"],
                x["evidence_score"],
                x["opportunity_score"],
                x["similarity"],
            ),
            reverse=True,
        )
        deduped_loop_results = []
        for annotated in annotated_loop_results:
            deduped_loop_results.append(annotated)
        clustered_loop_results, cluster_nms_stats = self._cluster_ranked_loop_windows(
            deduped_loop_results
        )
        print_timing("da3.streaming_loop.process_loop_list", process_loop_list_started)

        min_chunk_gap = max(int(min_chunk_gap), 0)
        min_frame_gap = max(int(min_frame_gap), 0)
        max_new_windows = max(int(max_new_windows), 0)
        max_serialized_windows = max(
            int(self.config["Loop"]["SALAD"].get("max_serialized_windows", 500)),
            0,
        )
        correction_chunk_pair_top1 = bool(
            self.config["Loop"]["SALAD"].get("correction_chunk_pair_top1", True)
        )
        correction_min_priority = max(
            float(self.config["Loop"]["SALAD"].get("correction_min_priority", 0.0)),
            0.0,
        )
        ranked_loop_window_count = len(deduped_loop_results)
        clustered_loop_window_count = len(clustered_loop_results)
        exported_loop_results = (
            clustered_loop_results[:max_serialized_windows]
            if max_serialized_windows
            else clustered_loop_results
        )
        serializable_windows = []
        new_windows = []
        rejected_windows = []
        correction_pair_suppressed_count = 0
        correction_priority_suppressed_count = 0
        for annotated in exported_loop_results:
            chunk_gap = int(annotated["chunk_gap"])
            frame_gap = int(annotated["frame_gap"])
            if chunk_gap < min_chunk_gap or frame_gap < min_frame_gap:
                rejected_windows.append(
                    {
                        "chunk_a": int(annotated["chunk_a"]),
                        "range_a": [int(annotated["range_a"][0]), int(annotated["range_a"][1])],
                        "chunk_b": int(annotated["chunk_b"]),
                        "range_b": [int(annotated["range_b"][0]), int(annotated["range_b"][1])],
                        "chunk_gap": int(chunk_gap),
                        "frame_gap": int(frame_gap),
                        "similarity": float(annotated["similarity"]),
                        "source_pair": annotated["source_pair"],
                        "path_rotation_deg": float(annotated["path_rotation_deg"]),
                        "path_rotation_total_deg": float(annotated["path_rotation_total_deg"]),
                        "path_rotation_peak_deg": float(annotated["path_rotation_peak_deg"]),
                        "path_rotation_net_deg": float(annotated["path_rotation_net_deg"]),
                        "path_rotation_unwrapped_net_deg": float(annotated["path_rotation_unwrapped_net_deg"]),
                        "path_rotation_principal_net_deg": float(annotated["path_rotation_principal_net_deg"]),
                        "path_rotation_directionality": float(annotated["path_rotation_directionality"]),
                        "path_rotation_measured_steps": int(annotated["path_rotation_measured_steps"]),
                        "path_rotation_source": annotated["path_rotation_source"],
                        "rotation_class": annotated["rotation_class"],
                        "cluster_size": int(annotated.get("cluster_size", 1)),
                        "priority_score": float(annotated["priority_score"]),
                        "sim_score": float(annotated["sim_score"]),
                        "window_support_score": float(annotated["window_support_score"]),
                        "evidence_score": float(annotated["evidence_score"]),
                        "rot_score": float(annotated["rot_score"]),
                        "frame_gap_score": float(annotated["frame_gap_score"]),
                        "chunk_gap_score": float(annotated["chunk_gap_score"]),
                        "one_way_score": float(annotated["one_way_score"]),
                        "full_turn_score": float(annotated["full_turn_score"]),
                        "opportunity_score": float(annotated["opportunity_score"]),
                        "opportunity_boost": float(annotated["opportunity_boost"]),
                        "reject_reason": "below_min_gap",
                    }
                )
                continue
            key = (
                int(annotated["chunk_a"]),
                int(annotated["range_a"][0]),
                int(annotated["range_a"][1]),
                int(annotated["chunk_b"]),
                int(annotated["range_b"][0]),
                int(annotated["range_b"][1]),
            )
            correction_pair_key = tuple(
                sorted((int(annotated["chunk_a"]), int(annotated["chunk_b"])))
            )
            is_new_window = key not in self.streaming_loop_detection_seen
            correction_pair_unused = (
                correction_pair_key not in self.streaming_loop_correction_seen_chunk_pairs
            )
            window = {
                "chunk_a": int(annotated["chunk_a"]),
                "range_a": [int(annotated["range_a"][0]), int(annotated["range_a"][1])],
                "chunk_b": int(annotated["chunk_b"]),
                "range_b": [int(annotated["range_b"][0]), int(annotated["range_b"][1])],
                "center_a": int(annotated["center_a"]),
                "center_b": int(annotated["center_b"]),
                "chunk_gap": int(chunk_gap),
                "frame_gap": int(frame_gap),
                "similarity": float(annotated["similarity"]),
                "source_pair": annotated["source_pair"],
                "path_rotation_deg": float(annotated["path_rotation_deg"]),
                "path_rotation_total_deg": float(annotated["path_rotation_total_deg"]),
                "path_rotation_peak_deg": float(annotated["path_rotation_peak_deg"]),
                "path_rotation_net_deg": float(annotated["path_rotation_net_deg"]),
                "path_rotation_unwrapped_net_deg": float(annotated["path_rotation_unwrapped_net_deg"]),
                "path_rotation_principal_net_deg": float(annotated["path_rotation_principal_net_deg"]),
                "path_rotation_directionality": float(annotated["path_rotation_directionality"]),
                "path_rotation_measured_steps": int(annotated["path_rotation_measured_steps"]),
                "path_rotation_source": annotated["path_rotation_source"],
                "rotation_class": annotated["rotation_class"],
                "cluster_size": int(annotated.get("cluster_size", 1)),
                "priority_score": float(annotated["priority_score"]),
                "sim_score": float(annotated["sim_score"]),
                "window_support_score": float(annotated["window_support_score"]),
                "evidence_score": float(annotated["evidence_score"]),
                "rot_score": float(annotated["rot_score"]),
                "frame_gap_score": float(annotated["frame_gap_score"]),
                "chunk_gap_score": float(annotated["chunk_gap_score"]),
                "one_way_score": float(annotated["one_way_score"]),
                "full_turn_score": float(annotated["full_turn_score"]),
                "opportunity_score": float(annotated["opportunity_score"]),
                "opportunity_boost": float(annotated["opportunity_boost"]),
                "is_new": bool(is_new_window),
                "correction_pair_key": [int(correction_pair_key[0]), int(correction_pair_key[1])],
                "correction_pair_unused": bool(correction_pair_unused),
                "correction_chunk_pair_top1": bool(correction_chunk_pair_top1),
            }
            serializable_windows.append(window)
            if is_new_window:
                if correction_min_priority and float(annotated["priority_score"]) < correction_min_priority:
                    correction_priority_suppressed_count += 1
                    continue
                if correction_chunk_pair_top1 and not correction_pair_unused:
                    correction_pair_suppressed_count += 1
                    self.streaming_loop_detection_seen.add(key)
                    continue
                if max_new_windows and len(new_windows) >= max_new_windows:
                    continue
                new_windows.append(window)
                self.streaming_loop_detection_seen.add(key)
                self.streaming_loop_correction_seen_chunk_pairs.add(correction_pair_key)

        loop_pairs = [[int(a), int(b)] for a, b, _ in self.loop_list]
        raw_loop_pairs = getattr(self.loop_detector, "raw_loop_closures", None) or []
        loop_nms_stats = getattr(self.loop_detector, "nms_stats", None) or {}
        rotation_class_counts = {}
        for window in serializable_windows:
            rotation_class = str(window.get("rotation_class") or "unknown")
            rotation_class_counts[rotation_class] = rotation_class_counts.get(rotation_class, 0) + 1
        info = {
            "enabled": True,
            "chunk_idx": int(chunk_idx),
            "processed_chunk_count": len(self.chunk_indices),
            "image_count": len(self.img_list or []),
            "raw_loop_pair_count": len(raw_loop_pairs),
            "loop_pair_count": len(loop_pairs),
            "loop_nms_stats": loop_nms_stats,
            "rotation_aware_priority": True,
            "rotation_class_counts": rotation_class_counts,
            "loop_pairs": loop_pairs,
            "ranked_loop_window_count": ranked_loop_window_count,
            "clustered_loop_window_count": clustered_loop_window_count,
            "cluster_nms_stats": cluster_nms_stats,
            "loop_window_count": len(serializable_windows),
            "serialized_loop_window_count": len(serializable_windows),
            "unserialized_loop_window_count": max(
                clustered_loop_window_count - len(exported_loop_results),
                0,
            ),
            "new_loop_window_count": len(new_windows),
            "correction_chunk_pair_top1": correction_chunk_pair_top1,
            "correction_min_priority": correction_min_priority,
            "correction_seen_chunk_pair_count": len(self.streaming_loop_correction_seen_chunk_pairs),
            "correction_pair_suppressed_count": correction_pair_suppressed_count,
            "correction_priority_suppressed_count": correction_priority_suppressed_count,
            "rejected_loop_window_count": len(rejected_windows),
            "rejected_loop_windows": rejected_windows,
            "min_chunk_gap": min_chunk_gap,
            "min_frame_gap": min_frame_gap,
            "max_new_windows": max_new_windows,
            "max_serialized_windows": max_serialized_windows,
            "loop_windows": serializable_windows,
            "new_loop_windows": new_windows,
            "wall_sec": print_timing("da3.streaming_loop.total", loop_detection_started),
        }
        self.streaming_loop_detection_history.append(info)
        output_path = os.path.join(self.output_dir, f"streaming_loop_candidates_chunk_{int(chunk_idx)}.json")
        with open(output_path, "w") as handle:
            json.dump(info, handle, indent=2)
            handle.write("\n")
        info["artifact"] = output_path
        return info

    def apply_streaming_loop_correction(self, loop_windows, chunk_idx):
        correction_started = timing_now()
        if not loop_windows:
            return None

        new_loop_predict_list = []
        for window in loop_windows:
            item = (
                int(window["chunk_a"]),
                (int(window["range_a"][0]), int(window["range_a"][1])),
                int(window["chunk_b"]),
                (int(window["range_b"][0]), int(window["range_b"][1])),
            )
            loop_window_started = timing_now()
            single_chunk_predictions = self.process_single_chunk(
                item[1],
                range_2=item[3],
                is_loop=True,
            )
            new_loop_predict_list.append((item, single_chunk_predictions))
            self.loop_predict_list.append((item, single_chunk_predictions))
            print_timing(
                f"da3.streaming_loop.window_reinference[{item[0]}->{item[2]}]",
                loop_window_started,
            )

        get_loop_sim3_started = timing_now()
        new_loop_sim3 = self.get_loop_sim3_from_loop_predict(new_loop_predict_list)
        self.loop_sim3_list.extend(new_loop_sim3)
        print_timing("da3.streaming_loop.get_new_loop_sim3", get_loop_sim3_started)

        input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)
        optimize_started = timing_now()
        self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
        self.streaming_frame_rotation_cache = {}
        self.streaming_chunk_rotation_cache = {}
        optimize_sec = print_timing("da3.streaming_loop.optimize", optimize_started)
        optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)
        self.streaming_map_epoch += 1
        self.plot_loop_closure(
            input_abs_poses,
            optimized_abs_poses,
            save_name=f"streaming_sim3_opt_epoch_{self.streaming_map_epoch}.png",
        )

        info = {
            "enabled": True,
            "chunk_idx": int(chunk_idx),
            "map_epoch": int(self.streaming_map_epoch),
            "new_loop_window_count": len(loop_windows),
            "new_loop_sim3_count": len(new_loop_sim3),
            "total_loop_sim3_count": len(self.loop_sim3_list),
            "optimized_sequential_transform_count": len(self.sim3_list),
            "optimize_sec": optimize_sec,
            "wall_sec": print_timing("da3.streaming_loop.correction_total", correction_started),
        }
        output_path = os.path.join(
            self.output_dir,
            f"streaming_loop_correction_epoch_{self.streaming_map_epoch}_chunk_{int(chunk_idx)}.json",
        )
        with open(output_path, "w") as handle:
            json.dump(info, handle, indent=2)
            handle.write("\n")
        info["artifact"] = output_path
        return info

    def save_depth_conf_result(self, predictions, chunk_idx, s, R, T):
        save_depth_conf_started = timing_now()
        if not self.config["Model"]["save_depth_conf_result"]:
            return
        os.makedirs(self.result_output_dir, exist_ok=True)

        chunk_start, chunk_end = self.chunk_indices[chunk_idx]

        if chunk_idx == 0:
            save_indices = list(range(0, chunk_end - chunk_start - self.overlap_e))
        elif chunk_idx == len(self.chunk_indices) - 1:
            save_indices = list(range(self.overlap_s, chunk_end - chunk_start))
        else:
            save_indices = list(range(self.overlap_s, chunk_end - chunk_start - self.overlap_e))

        print("[save_depth_conf_result] save_indices:")
        for local_idx in save_indices:
            global_idx = chunk_start + local_idx
            print(f"{global_idx}, ", end="")
        print("")

        if not save_indices:
            print_timing("da3.save_depth_conf_result.total", save_depth_conf_started)
            return

        save_debug_info = self.config["Model"]["save_debug_info"]
        configured_max_workers = self.config["Model"].get("save_depth_conf_max_workers")
        if configured_max_workers is None:
            max_workers = min(len(save_indices), max(effective_cpu_count() - 1, 1))
        else:
            max_workers = min(len(save_indices), max(int(configured_max_workers), 1))

        print(f"[save_depth_conf_result] max_workers={max_workers}")

        def _save_one(local_idx):
            global_idx = chunk_start + local_idx
            image = predictions.processed_images[local_idx]  # [H, W, 3] uint8
            depth = predictions.depth[local_idx]  # [H, W] float32
            conf = predictions.conf[local_idx]  # [H, W] float32
            intrinsics = predictions.intrinsics[local_idx]  # [3, 3] float32

            filename = f"frame_{global_idx}.npz"
            filepath = os.path.join(self.result_output_dir, filename)

            if save_debug_info:
                np.savez_compressed(
                    filepath,
                    image=image,
                    depth=depth,
                    conf=conf,
                    intrinsics=intrinsics,
                    extrinsics=predictions.extrinsics[local_idx],
                    s=s,
                    R=R,
                    T=T,
                )
            else:
                np.savez_compressed(
                    filepath, image=image, depth=depth, conf=conf, intrinsics=intrinsics
                )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_save_one, save_indices))

        print_timing("da3.save_depth_conf_result.total", save_depth_conf_started)

    def _pointcloud_conf_threshold(self, confs):
        save_cfg = self.config["Model"]["Pointcloud_Save"]
        return float(np.mean(confs) * save_cfg["conf_threshold_coef"])

    def _load_unaligned_chunk(self, chunk_idx):
        return np.load(
            os.path.join(self.result_unaligned_dir, f"chunk_{int(chunk_idx)}.npy"),
            allow_pickle=True,
        ).item()

    def _streaming_chunk_world_points(self, chunk_idx, chunk_data, accumulated_sim3):
        if chunk_idx == 0:
            return depth_to_point_cloud_vectorized(
                chunk_data.depth,
                chunk_data.intrinsics,
                chunk_data.extrinsics,
            )

        s, R, t = accumulated_sim3[chunk_idx - 1]
        world_points = depth_to_point_cloud_optimized_torch(
            chunk_data.depth,
            chunk_data.intrinsics,
            chunk_data.extrinsics,
        )
        return apply_sim3_direct_torch(world_points, s, R, t)

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        chunk_total_started = timing_now()
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        # images = load_and_preprocess_images(chunk_image_paths).to(self.device)
        print(f"Loaded {len(chunk_image_paths)} images")

        ref_view_strategy = self.config["Model"][
            "ref_view_strategy" if not is_loop else "ref_view_strategy_loop"
        ]

        inference_started = timing_now()
        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images = chunk_image_paths
                # images: ['xxx.png', 'xxx.png', ...]

                predictions = self.model.inference(images, ref_view_strategy=ref_view_strategy)

                predictions.depth = np.squeeze(predictions.depth)
                predictions.conf -= 1.0

                print(predictions.processed_images.shape)  # [N, H, W, 3] uint8
                print(predictions.depth.shape)  # [N, H, W] float32
                print(predictions.conf.shape)  # [N, H, W] float32
                print(predictions.extrinsics.shape)  # [N, 3, 4] float32 (w2c)
                print(predictions.intrinsics.shape)  # [N, 3, 3] float32
        torch.cuda.empty_cache()
        print_timing("da3.process_single_chunk.inference_total", inference_started)

        # Save predictions to disk instead of keeping in memory
        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"

        save_path = os.path.join(save_dir, filename)

        if not is_loop and range_2 is None:
            extrinsics = predictions.extrinsics
            intrinsics = predictions.intrinsics
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))

        save_predictions_started = timing_now()
        np.save(save_path, predictions)
        print_timing("da3.process_single_chunk.save_predictions", save_predictions_started)

        phase = "loop" if is_loop else "chunk"
        print_timing(f"da3.process_single_chunk.total[{phase}]", chunk_total_started)
        return predictions

    def get_chunk_indices(self):
        if len(self.img_list) <= self.chunk_size:
            num_chunks = 1
            chunk_indices = [(0, len(self.img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                chunk_indices.append((start_idx, end_idx))
        return chunk_indices, num_chunks

    def align_2pcds(
        self,
        point_map1,
        conf1,
        point_map2,
        conf2,
        chunk1_depth,
        chunk2_depth,
        chunk1_depth_conf,
        chunk2_depth_conf,
    ):

        conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1

        scale_factor = None
        if self.config["Model"]["align_method"] == "scale+se3":
            scale_factor_return, quality_score, method_used = precompute_scale_chunks_with_depth(
                chunk1_depth,
                chunk1_depth_conf,
                chunk2_depth,
                chunk2_depth_conf,
                method=self.config["Model"]["scale_compute_method"],
            )
            print(
                f"[Depth Scale Precompute] scale: {scale_factor_return}, \
                    quality_score: {quality_score}, method_used: {method_used}"
            )
            scale_factor = scale_factor_return

        s, R, t = weighted_align_point_maps(
            point_map1,
            conf1,
            point_map2,
            conf2,
            conf_threshold=conf_threshold,
            config=self.config,
            precompute_scale=scale_factor,
        )
        print("Estimated Scale:", s)
        print("Estimated Rotation:\n", R)
        print("Estimated Translation:", t)

        return s, R, t

    def get_loop_sim3_from_loop_predict(self, loop_predict_list):
        loop_sim3_list = []
        for item in loop_predict_list:
            chunk_idx_a = item[0][0]
            chunk_idx_b = item[0][2]
            chunk_a_range = item[0][1]
            chunk_b_range = item[0][3]

            point_map_loop_org = depth_to_point_cloud_vectorized(
                item[1].depth, item[1].intrinsics, item[1].extrinsics
            )

            chunk_a_s = 0
            chunk_a_e = chunk_a_len = chunk_a_range[1] - chunk_a_range[0]
            chunk_b_s = -chunk_b_range[1] + chunk_b_range[0]
            chunk_b_e = point_map_loop_org.shape[0]
            chunk_b_len = chunk_b_range[1] - chunk_b_range[0]

            chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
            chunk_a_rela_end = chunk_a_rela_begin + chunk_a_len
            chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
            chunk_b_rela_end = chunk_b_rela_begin + chunk_b_len

            print("chunk_a align")

            point_map_loop_a = point_map_loop_org[chunk_a_s:chunk_a_e]
            conf_loop = item[1].conf[chunk_a_s:chunk_a_e]
            print(self.chunk_indices[chunk_idx_a])
            print(chunk_a_range)
            print(chunk_a_rela_begin, chunk_a_rela_end)
            chunk_data_a = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"),
                allow_pickle=True,
            ).item()

            point_map_a = depth_to_point_cloud_vectorized(
                chunk_data_a.depth, chunk_data_a.intrinsics, chunk_data_a.extrinsics
            )
            point_map_a = point_map_a[chunk_a_rela_begin:chunk_a_rela_end]
            conf_a = chunk_data_a.conf[chunk_a_rela_begin:chunk_a_rela_end]

            if self.config["Model"]["align_method"] == "scale+se3":
                chunk_a_depth = np.squeeze(chunk_data_a.depth[chunk_a_rela_begin:chunk_a_rela_end])
                chunk_a_depth_conf = np.squeeze(
                    chunk_data_a.conf[chunk_a_rela_begin:chunk_a_rela_end]
                )
                chunk_a_loop_depth = np.squeeze(item[1].depth[chunk_a_s:chunk_a_e])
                chunk_a_loop_depth_conf = np.squeeze(item[1].conf[chunk_a_s:chunk_a_e])
            else:
                chunk_a_depth = None
                chunk_a_loop_depth = None
                chunk_a_depth_conf = None
                chunk_a_loop_depth_conf = None

            s_a, R_a, t_a = self.align_2pcds(
                point_map_a,
                conf_a,
                point_map_loop_a,
                conf_loop,
                chunk_a_depth,
                chunk_a_loop_depth,
                chunk_a_depth_conf,
                chunk_a_loop_depth_conf,
            )

            print("chunk_b align")

            point_map_loop_b = point_map_loop_org[chunk_b_s:chunk_b_e]
            conf_loop = item[1].conf[chunk_b_s:chunk_b_e]
            print(self.chunk_indices[chunk_idx_b])
            print(chunk_b_range)
            print(chunk_b_rela_begin, chunk_b_rela_end)
            chunk_data_b = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"),
                allow_pickle=True,
            ).item()

            point_map_b = depth_to_point_cloud_vectorized(
                chunk_data_b.depth, chunk_data_b.intrinsics, chunk_data_b.extrinsics
            )
            point_map_b = point_map_b[chunk_b_rela_begin:chunk_b_rela_end]
            conf_b = chunk_data_b.conf[chunk_b_rela_begin:chunk_b_rela_end]

            if self.config["Model"]["align_method"] == "scale+se3":
                chunk_b_depth = np.squeeze(chunk_data_b.depth[chunk_b_rela_begin:chunk_b_rela_end])
                chunk_b_depth_conf = np.squeeze(
                    chunk_data_b.conf[chunk_b_rela_begin:chunk_b_rela_end]
                )
                chunk_b_loop_depth = np.squeeze(item[1].depth[chunk_b_s:chunk_b_e])
                chunk_b_loop_depth_conf = np.squeeze(item[1].conf[chunk_b_s:chunk_b_e])
            else:
                chunk_b_depth = None
                chunk_b_loop_depth = None
                chunk_b_depth_conf = None
                chunk_b_loop_depth_conf = None

            s_b, R_b, t_b = self.align_2pcds(
                point_map_b,
                conf_b,
                point_map_loop_b,
                conf_loop,
                chunk_b_depth,
                chunk_b_loop_depth,
                chunk_b_depth_conf,
                chunk_b_loop_depth_conf,
            )

            print("a -> b SIM 3")
            s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
            print("Estimated Scale:", s_ab)
            print("Estimated Rotation:\n", R_ab)
            print("Estimated Translation:", t_ab)

            loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

        return loop_sim3_list

    def plot_loop_closure(
        self, input_abs_poses, optimized_abs_poses, save_name="sim3_opt_result.png"
    ):
        def extract_xyz(pose_tensor):
            poses = pose_tensor.cpu().numpy()
            return poses[:, 0], poses[:, 1], poses[:, 2]

        x0, _, y0 = extract_xyz(input_abs_poses)
        x1, _, y1 = extract_xyz(optimized_abs_poses)

        # Visual in png format
        plt.figure(figsize=(8, 6))
        plt.plot(x0, y0, "o--", alpha=0.45, label="Before Optimization")
        plt.plot(x1, y1, "o-", label="After Optimization")
        for i, j, _ in self.loop_sim3_list:
            plt.plot(
                [x0[i], x0[j]],
                [y0[i], y0[j]],
                "r--",
                alpha=0.25,
                label="Loop (Before)" if i == 5 else "",
            )
            plt.plot(
                [x1[i], x1[j]],
                [y1[i], y1[j]],
                "g-",
                alpha=0.25,
                label="Loop (After)" if i == 5 else "",
            )
        plt.gca().set_aspect("equal")
        plt.title("Sim3 Loop Closure Optimization")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def process_long_sequence(self):
        process_long_sequence_started = timing_now()
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"[SETTING ERROR] Overlap ({self.overlap}) \
                    must be less than chunk size ({self.chunk_size})"
            )

        self.chunk_indices, num_chunks = self.get_chunk_indices()

        print(
            f"Processing {len(self.img_list)} images in {num_chunks} \
                chunks of size {self.chunk_size} with {self.overlap} overlap"
        )

        pre_predictions = None
        for chunk_idx in range(len(self.chunk_indices)):
            print(f"[Progress]: {chunk_idx}/{len(self.chunk_indices)}")
            cur_predictions = self.process_single_chunk(
                self.chunk_indices[chunk_idx], chunk_idx=chunk_idx
            )
            torch.cuda.empty_cache()

            if chunk_idx > 0:
                print(
                    f"Aligning {chunk_idx-1} and {chunk_idx} (Total {len(self.chunk_indices)-1})"
                )
                chunk_data1 = pre_predictions
                chunk_data2 = cur_predictions

                point_map1 = depth_to_point_cloud_vectorized(
                    chunk_data1.depth, chunk_data1.intrinsics, chunk_data1.extrinsics
                )
                point_map2 = depth_to_point_cloud_vectorized(
                    chunk_data2.depth, chunk_data2.intrinsics, chunk_data2.extrinsics
                )

                point_map1 = point_map1[-self.overlap :]
                point_map2 = point_map2[: self.overlap]
                conf1 = chunk_data1.conf[-self.overlap :]
                conf2 = chunk_data2.conf[: self.overlap]

                if self.config["Model"]["align_method"] == "scale+se3":
                    chunk1_depth = np.squeeze(chunk_data1.depth[-self.overlap :])
                    chunk2_depth = np.squeeze(chunk_data2.depth[: self.overlap])
                    chunk1_depth_conf = np.squeeze(chunk_data1.conf[-self.overlap :])
                    chunk2_depth_conf = np.squeeze(chunk_data2.conf[: self.overlap])
                else:
                    chunk1_depth = None
                    chunk2_depth = None
                    chunk1_depth_conf = None
                    chunk2_depth_conf = None

                sequential_align_started = timing_now()
                s, R, t = self.align_2pcds(
                    point_map1,
                    conf1,
                    point_map2,
                    conf2,
                    chunk1_depth,
                    chunk2_depth,
                    chunk1_depth_conf,
                    chunk2_depth_conf,
                )
                print_timing(
                    f"da3.process_long_sequence.sequential_chunk_align[{chunk_idx-1}->{chunk_idx}]",
                    sequential_align_started,
                )
                self.sim3_list.append((s, R, t))

            pre_predictions = cur_predictions

        if self.loop_enable:
            loop_total_started = timing_now()
            self.loop_list = self.get_loop_pairs()
            self.release_loop_detector_session()  # Save GPU memory while preserving runtime assets.

            torch.cuda.empty_cache()

            print("Loop SIM(3) estimating...")
            process_loop_list_started = timing_now()
            loop_results = process_loop_list(
                self.chunk_indices,
                self.loop_list,
                half_window=int(self.config["Model"]["loop_chunk_size"] / 2),
            )
            loop_results = remove_duplicates(loop_results)
            print_timing("da3.loop.process_loop_list", process_loop_list_started)
            print(loop_results)
            # return e.g. (31, (1574, 1594), 2, (129, 149))
            loop_window_reinference_started = timing_now()
            for item in loop_results:
                single_chunk_predictions = self.process_single_chunk(
                    item[1], range_2=item[3], is_loop=True
                )

                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)

            print_timing("da3.loop.loop_window_reinference_total", loop_window_reinference_started)
            get_loop_sim3_started = timing_now()
            self.loop_sim3_list = self.get_loop_sim3_from_loop_predict(self.loop_predict_list)
            print_timing("da3.loop.get_loop_sim3_from_loop_predict", get_loop_sim3_started)

            sequential_to_absolute_started = timing_now()
            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(
                self.sim3_list
            )  # just for plot
            print_timing("da3.loop.sequential_to_absolute_before_opt", sequential_to_absolute_started)
            optimize_started = timing_now()
            self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
            print_timing("da3.loop.optimize", optimize_started)
            optimized_absolute_started = timing_now()
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(
                self.sim3_list
            )  # just for plot

            print_timing("da3.loop.sequential_to_absolute_after_opt", optimized_absolute_started)
            plot_loop_closure_started = timing_now()
            self.plot_loop_closure(
                input_abs_poses, optimized_abs_poses, save_name="sim3_opt_result.png"
            )
            print_timing("da3.loop.plot_loop_closure", plot_loop_closure_started)
            print_timing("da3.loop.total", loop_total_started)

        apply_alignment_started = timing_now()
        print("Apply alignment")
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)

        if len(self.chunk_indices) == 1:
            chunk_data_first = np.load(
                os.path.join(self.result_unaligned_dir, "chunk_0.npy"), allow_pickle=True
            ).item()
            np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)
            points_first = depth_to_point_cloud_vectorized(
                chunk_data_first.depth,
                chunk_data_first.intrinsics,
                chunk_data_first.extrinsics,
            )
            colors_first = chunk_data_first.processed_images
            confs_first = chunk_data_first.conf
            ply_path_first = os.path.join(self.pcd_dir, "0_pcd.ply")
            save_confident_pointcloud_batch(
                points=points_first,
                colors=colors_first,
                confs=confs_first,
                output_path=ply_path_first,
                conf_threshold=np.mean(confs_first)
                * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
            )
            if self.config["Model"]["save_depth_conf_result"]:
                predictions = chunk_data_first
                self.save_depth_conf_result(predictions, 0, 1, np.eye(3), np.array([0, 0, 0]))
            save_camera_poses_started = timing_now()
            self.save_camera_poses()
            print_timing("da3.save_camera_poses", save_camera_poses_started)
            print_timing("da3.apply_alignment_and_export_total", apply_alignment_started)
            print("Done.")
            print_timing("da3.process_long_sequence.total", process_long_sequence_started)
            return
        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(f"Applying {chunk_idx+1} -> {chunk_idx} (Total {len(self.chunk_indices)-1})")
            s, R, t = self.sim3_list[chunk_idx]

            chunk_data = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"),
                allow_pickle=True,
            ).item()

            aligned_chunk_data = {}

            aligned_chunk_data["world_points"] = depth_to_point_cloud_optimized_torch(
                chunk_data.depth, chunk_data.intrinsics, chunk_data.extrinsics
            )
            aligned_chunk_data["world_points"] = apply_sim3_direct_torch(
                aligned_chunk_data["world_points"], s, R, t
            )

            aligned_chunk_data["conf"] = chunk_data.conf
            aligned_chunk_data["images"] = chunk_data.processed_images

            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy")
            np.save(aligned_path, aligned_chunk_data)

            if chunk_idx == 0:
                chunk_data_first = np.load(
                    os.path.join(self.result_unaligned_dir, "chunk_0.npy"), allow_pickle=True
                ).item()
                np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)
                points_first = depth_to_point_cloud_vectorized(
                    chunk_data_first.depth,
                    chunk_data_first.intrinsics,
                    chunk_data_first.extrinsics,
                )
                colors_first = chunk_data_first.processed_images
                confs_first = chunk_data_first.conf
                ply_path_first = os.path.join(self.pcd_dir, "0_pcd.ply")
                save_confident_pointcloud_batch(
                    points=points_first,  # shape: (H, W, 3)
                    colors=colors_first,  # shape: (H, W, 3)
                    confs=confs_first,  # shape: (H, W)
                    output_path=ply_path_first,
                    conf_threshold=np.mean(confs_first)
                    * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                    sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
                )
                if self.config["Model"]["save_depth_conf_result"]:
                    predictions = chunk_data_first
                    self.save_depth_conf_result(predictions, 0, 1, np.eye(3), np.array([0, 0, 0]))

            points = aligned_chunk_data["world_points"].reshape(-1, 3)
            colors = (aligned_chunk_data["images"].reshape(-1, 3)).astype(np.uint8)
            confs = aligned_chunk_data["conf"].reshape(-1)
            ply_path = os.path.join(self.pcd_dir, f"{chunk_idx+1}_pcd.ply")
            save_confident_pointcloud_batch(
                points=points,  # shape: (H, W, 3)
                colors=colors,  # shape: (H, W, 3)
                confs=confs,  # shape: (H, W)
                output_path=ply_path,
                conf_threshold=np.mean(confs)
                * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
            )

            if self.config["Model"]["save_depth_conf_result"]:
                predictions = chunk_data
                predictions.depth *= s
                self.save_depth_conf_result(predictions, chunk_idx + 1, s, R, t)

        save_camera_poses_started = timing_now()
        self.save_camera_poses()
        print_timing("da3.save_camera_poses", save_camera_poses_started)
        print_timing("da3.apply_alignment_and_export_total", apply_alignment_started)

        print("Done.")
        print_timing("da3.process_long_sequence.total", process_long_sequence_started)

    def process_streaming_chunk(
        self,
        image_paths,
        chunk_range,
        chunk_idx,
        enable_loop_detection=False,
        loop_start_chunk=1,
        enable_loop_correction=False,
        loop_min_chunk_gap=2,
        loop_min_frame_gap=0,
        loop_max_new_windows=0,
        loop_detection_interval=1,
        loop_correction_min_new_windows=2,
        reexport_corrected_chunks=True,
    ):
        streaming_chunk_started = timing_now()
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"[SETTING ERROR] Overlap ({self.overlap}) \
                    must be less than chunk size ({self.chunk_size})"
            )

        self.img_list = list(image_paths)
        if self.chunk_indices is None:
            self.chunk_indices = []
        chunk_range = (int(chunk_range[0]), int(chunk_range[1]))
        chunk_idx = int(chunk_idx)
        if chunk_idx != len(self.chunk_indices):
            raise ValueError(
                f"streaming chunks must arrive in order: got {chunk_idx}, "
                f"expected {len(self.chunk_indices)}"
            )
        self.chunk_indices.append(chunk_range)

        cur_predictions = self.process_single_chunk(chunk_range, chunk_idx=chunk_idx)
        torch.cuda.empty_cache()

        if chunk_idx > 0:
            print(f"Aligning {chunk_idx-1} and {chunk_idx}")
            chunk_data1 = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx-1}.npy"),
                allow_pickle=True,
            ).item()
            chunk_data2 = cur_predictions

            point_map1 = depth_to_point_cloud_vectorized(
                chunk_data1.depth, chunk_data1.intrinsics, chunk_data1.extrinsics
            )
            point_map2 = depth_to_point_cloud_vectorized(
                chunk_data2.depth, chunk_data2.intrinsics, chunk_data2.extrinsics
            )

            point_map1 = point_map1[-self.overlap :]
            point_map2 = point_map2[: self.overlap]
            conf1 = chunk_data1.conf[-self.overlap :]
            conf2 = chunk_data2.conf[: self.overlap]

            if self.config["Model"]["align_method"] == "scale+se3":
                chunk1_depth = np.squeeze(chunk_data1.depth[-self.overlap :])
                chunk2_depth = np.squeeze(chunk_data2.depth[: self.overlap])
                chunk1_depth_conf = np.squeeze(chunk_data1.conf[-self.overlap :])
                chunk2_depth_conf = np.squeeze(chunk_data2.conf[: self.overlap])
            else:
                chunk1_depth = None
                chunk2_depth = None
                chunk1_depth_conf = None
                chunk2_depth_conf = None

            sequential_align_started = timing_now()
            s, R, t = self.align_2pcds(
                point_map1,
                conf1,
                point_map2,
                conf2,
                chunk1_depth,
                chunk2_depth,
                chunk1_depth_conf,
                chunk2_depth_conf,
            )
            print_timing(
                f"da3.streaming_chunk.sequential_chunk_align[{chunk_idx-1}->{chunk_idx}]",
                sequential_align_started,
            )
            self.sim3_list.append((s, R, t))

        loop_info = None
        correction_info = None
        loop_detection_interval = max(int(loop_detection_interval), 1)
        should_run_loop_detection = (
            enable_loop_detection
            and self.loop_enable
            and chunk_idx >= int(loop_start_chunk)
            and chunk_idx % loop_detection_interval == 0
        )
        if should_run_loop_detection:
            loop_info = self.detect_streaming_loop_candidates(
                chunk_idx,
                min_chunk_gap=loop_min_chunk_gap,
                min_frame_gap=loop_min_frame_gap,
                max_new_windows=loop_max_new_windows,
            )
            new_loop_windows = loop_info.get("new_loop_windows") or []
            if enable_loop_correction and new_loop_windows:
                self.streaming_loop_correction_pending_windows.extend(new_loop_windows)
            if enable_loop_correction and self.streaming_loop_correction_pending_windows:
                min_new_windows = max(int(loop_correction_min_new_windows), 1)
                if len(self.streaming_loop_correction_pending_windows) >= min_new_windows:
                    pending_windows = list(self.streaming_loop_correction_pending_windows)
                    self.streaming_loop_correction_pending_windows = []
                    correction_info = self.apply_streaming_loop_correction(
                        pending_windows,
                        chunk_idx,
                    )
            if loop_info is not None:
                loop_info["pending_loop_window_count"] = len(self.streaming_loop_correction_pending_windows)
                loop_info["correction_min_new_windows"] = max(int(loop_correction_min_new_windows), 1)
                loop_info["loop_detection_interval"] = loop_detection_interval

        artifacts = self.export_streaming_chunk_artifacts(
            chunk_idx,
            map_epoch=self.streaming_map_epoch,
            map_mode="corrected" if self.streaming_map_epoch else "provisional",
        )
        if loop_info is not None:
            artifacts["streaming_loop_detection"] = loop_info
        if correction_info is not None:
            artifacts["streaming_loop_correction"] = correction_info
            if reexport_corrected_chunks:
                artifacts["streaming_loop_corrected_chunks"] = (
                    self.export_streaming_corrected_chunk_artifacts(correction_info["map_epoch"])
                )
        print_timing(
            f"da3.process_streaming_chunk.total[{chunk_idx}]",
            streaming_chunk_started,
        )
        return artifacts

    def export_streaming_chunk_artifacts(
        self,
        chunk_idx,
        *,
        accumulated_sim3=None,
        artifact_tag=None,
        map_epoch=None,
        map_mode="provisional",
    ):
        export_started = timing_now()
        chunk_idx = int(chunk_idx)
        chunk_data = self._load_unaligned_chunk(chunk_idx)

        if accumulated_sim3 is None:
            accumulated_sim3 = accumulate_sim3_transforms(list(self.sim3_list))
        path_tag = str(artifact_tag) if artifact_tag is not None else str(chunk_idx)

        world_points = self._streaming_chunk_world_points(
            chunk_idx,
            chunk_data,
            accumulated_sim3,
        )
        points = world_points.reshape(-1, 3)
        colors = (chunk_data.processed_images.reshape(-1, 3)).astype(np.uint8)
        confs = chunk_data.conf.reshape(-1)
        ply_path = os.path.join(self.pcd_dir, f"{path_tag}_pcd.ply")
        save_confident_pointcloud_batch(
            points=points,
            colors=colors,
            confs=confs,
            output_path=ply_path,
            conf_threshold=self._pointcloud_conf_threshold(confs),
            sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
        )

        camera_poses_path = os.path.join(self.output_dir, f"camera_poses_chunk_{path_tag}.txt")
        intrinsic_path = os.path.join(self.output_dir, f"intrinsic_chunk_{path_tag}.txt")
        self.save_streaming_chunk_camera_files(
            chunk_idx,
            chunk_data,
            accumulated_sim3,
            camera_poses_path,
            intrinsic_path,
        )

        print_timing(f"da3.export_streaming_chunk_artifacts[{chunk_idx}]", export_started)
        return {
            "chunk_index": chunk_idx,
            "chunk_range": self.chunk_indices[chunk_idx],
            "pcd": ply_path,
            "camera_poses": camera_poses_path,
            "intrinsic": intrinsic_path,
            "processed_chunk_count": len(self.chunk_indices),
            "map_epoch": int(map_epoch) if map_epoch is not None else int(self.streaming_map_epoch),
            "map_mode": map_mode,
            "pointcloud_mode": "raw_confident_points",
            "requires_map_replace": False,
        }

    def export_streaming_corrected_chunk_artifacts(self, map_epoch):
        corrected_started = timing_now()
        accumulated_sim3 = accumulate_sim3_transforms(list(self.sim3_list))
        corrected_chunks = []
        for chunk_idx in range(len(self.chunk_indices)):
            corrected_chunks.append(
                self.export_streaming_chunk_artifacts(
                    chunk_idx,
                    accumulated_sim3=accumulated_sim3,
                    artifact_tag=f"epoch_{int(map_epoch)}_chunk_{chunk_idx}",
                    map_epoch=map_epoch,
                    map_mode="corrected",
                )
            )
        print_timing("da3.streaming_loop.export_corrected_chunks", corrected_started)
        return corrected_chunks

    def save_streaming_chunk_camera_files(
        self,
        chunk_idx,
        chunk_data,
        accumulated_sim3,
        camera_poses_path,
        intrinsic_path,
    ):
        chunk_idx = int(chunk_idx)
        sim3 = accumulated_sim3[chunk_idx - 1] if chunk_idx > 0 else None
        if sim3 is not None:
            s, R, t = sim3
            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

        with open(camera_poses_path, "w") as pose_file, open(intrinsic_path, "w") as intrinsic_file:
            for local_idx in range(len(chunk_data.extrinsics)):
                w2c = np.eye(4)
                w2c[:3, :] = chunk_data.extrinsics[local_idx]
                c2w = np.linalg.inv(w2c)
                if sim3 is not None:
                    c2w = S @ c2w
                    c2w[:3, :3] /= s
                flat_pose = c2w.flatten()
                pose_file.write(" ".join([str(x) for x in flat_pose]) + "\n")

                intrinsic = chunk_data.intrinsics[local_idx]
                fx = intrinsic[0, 0]
                fy = intrinsic[1, 1]
                cx = intrinsic[0, 2]
                cy = intrinsic[1, 2]
                intrinsic_file.write(f"{fx} {fy} {cx} {cy}\n")

    def run(self):
        run_started = timing_now()
        print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg"))
            + glob.glob(os.path.join(self.img_dir, "*.png"))
        )
        # print(self.img_list)
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        self.process_long_sequence()
        print_timing("da3.run.total", run_started)

    def release_loop_detector_session(self):
        loop_detector = getattr(self, "loop_detector", None)
        if loop_detector is None:
            return
        loop_detector.image_paths = None
        loop_detector.descriptors = None
        loop_detector.loop_closures = None
        if self._uses_preloaded_loop_detector:
            loop_detector.model = None
            loop_detector.device = None
        del self.loop_detector

    def reset_session_state(self):
        reset_started = timing_now()
        self.release_loop_detector_session()
        self.img_list = None
        self.chunk_indices = None
        self.all_camera_poses = []
        self.all_camera_intrinsics = []
        self.loop_list = []
        self.sim3_list = []
        self.loop_sim3_list = []
        self.loop_predict_list = []
        self.streaming_loop_detection_history = []
        self.streaming_loop_detection_seen = set()
        self.streaming_loop_correction_seen_chunk_pairs = set()
        self.streaming_loop_correction_pending_windows = []
        self.streaming_map_epoch = 0
        self.streaming_frame_rotation_cache = {}
        self.streaming_chunk_rotation_cache = {}
        self.skyseg_session = None
        if not self._uses_preloaded_model:
            self.model = None
        print_timing("da3.reset_session_state", reset_started)

    def save_camera_poses(self):
        save_camera_poses_started = timing_now()
        """
        Save camera poses from all chunks to txt and ply files
        - txt file: Each line contains a 4x4 C2W matrix flattened into 16 numbers
        - ply file: Camera poses visualized as points with different colors for each chunk
        """
        chunk_colors = [
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [128, 0, 0],  # Dark Red
            [0, 128, 0],  # Dark Green
            [0, 0, 128],  # Dark Blue
            [128, 128, 0],  # Olive
        ]
        print("Saving all camera poses to txt file...")

        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        _, first_chunk_intrinsics = self.all_camera_intrinsics[0]

        first_chunk_end = (
            first_chunk_range[1]
            if len(self.all_camera_poses) == 1
            else first_chunk_range[1] - self.overlap_e
        )
        for i, idx in enumerate(range(first_chunk_range[0], first_chunk_end)):
            w2c = np.eye(4)
            w2c[:3, :] = first_chunk_extrinsics[i]
            c2w = np.linalg.inv(w2c)
            all_poses[idx] = c2w
            all_intrinsics[idx] = first_chunk_intrinsics[i]

        for chunk_idx in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            _, chunk_intrinsics = self.all_camera_intrinsics[chunk_idx]
            s, R, t = self.sim3_list[
                chunk_idx - 1
            ]  # When call self.save_camera_poses(), all the sim3 are aligned to the first chunk.

            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            chunk_range_end = (
                chunk_range[1] - self.overlap_e
                if chunk_idx < len(self.all_camera_poses) - 1
                else chunk_range[1]
            )

            for i, idx in enumerate(range(chunk_range[0] + self.overlap_s, chunk_range_end)):
                w2c = np.eye(4)
                w2c[:3, :] = chunk_extrinsics[i + self.overlap_s]
                c2w = np.linalg.inv(w2c)

                transformed_c2w = S @ c2w  # Be aware of the left multiplication!
                transformed_c2w[:3, :3] /= s  # Normalize rotation

                all_poses[idx] = transformed_c2w
                all_intrinsics[idx] = chunk_intrinsics[i + self.overlap_s]

        poses_path = os.path.join(self.output_dir, "camera_poses.txt")
        with open(poses_path, "w") as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(" ".join([str(x) for x in flat_pose]) + "\n")

        print(f"Camera poses saved to {poses_path}")

        intrinsics_path = os.path.join(self.output_dir, "intrinsic.txt")
        with open(intrinsics_path, "w") as f:
            for intrinsic in all_intrinsics:
                fx = intrinsic[0, 0]
                fy = intrinsic[1, 1]
                cx = intrinsic[0, 2]
                cy = intrinsic[1, 2]
                f.write(f"{fx} {fy} {cx} {cy}\n")

        print(f"Camera intrinsics saved to {intrinsics_path}")

        ply_path = os.path.join(self.output_dir, "camera_poses.ply")
        with open(ply_path, "w") as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(all_poses)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                f.write(
                    f"{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n"
                )

        print(f"Camera poses visualization saved to {ply_path}")
        print_timing("da3.save_camera_poses.total", save_camera_poses_started)

    def close(self):
        close_started = timing_now()
        """
        Clean up temporary files and calculate reclaimed disk space.

        This method deletes all temporary files generated during processing from three directories:
        - Unaligned results
        - Aligned results
        - Loop results

        ~50 GiB for 4500-frame KITTI 00,
        ~35 GiB for 2700-frame KITTI 05,
        or ~5 GiB for 300-frame short seq.
        """
        if not self.delete_temp_files:
            return

        total_space = 0

        print(f"Deleting the temp files under {self.result_unaligned_dir}")
        for filename in os.listdir(self.result_unaligned_dir):
            file_path = os.path.join(self.result_unaligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f"Deleting the temp files under {self.result_aligned_dir}")
        for filename in os.listdir(self.result_aligned_dir):
            file_path = os.path.join(self.result_aligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f"Deleting the temp files under {self.result_loop_dir}")
        for filename in os.listdir(self.result_loop_dir):
            file_path = os.path.join(self.result_loop_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)
        print("Deleting temp files done.")

        print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")
        print_timing("da3.close.total", close_started)


def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)

        dst_path = os.path.join(dst_dir, os.path.basename(src_path))

        shutil.copy2(src_path, dst_path)
        print(f"config yaml file has been copied to: {dst_path}")
        return dst_path

    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as e:
        print(f"Copy Error: {e}")


def _default_save_dir(image_dir):
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = "./exps"
    return os.path.join(exp_dir, image_dir.replace("/", "_"), current_datetime)


def run_da3_job(
    image_dir,
    config_path="./configs/base_config.yaml",
    output_dir=None,
    runtime_assets=None,
    base_dir=None,
    absolute_paths=False,
):
    main_started = timing_now()

    load_config_started = timing_now()
    config, resolved_config_path = load_job_config(
        config_path, base_dir=base_dir, absolute_paths=absolute_paths
    )
    print_timing("da3.main.load_config", load_config_started)

    if absolute_paths:
        image_dir = _abs_path(image_dir, base_dir)
        if output_dir is not None:
            output_dir = _abs_path(output_dir, base_dir)

    if output_dir is not None:
        save_dir = output_dir
    else:
        save_dir = _default_save_dir(image_dir)
        if absolute_paths:
            save_dir = _abs_path(save_dir, base_dir)

    if not os.path.exists(save_dir):
        save_dir_prepare_started = timing_now()
        os.makedirs(save_dir)
        print(f"The exp will be saved under dir: {save_dir}")
        copy_file(resolved_config_path, save_dir)
        print_timing("da3.main.prepare_save_dir_and_copy_config", save_dir_prepare_started)

    if config["Model"]["align_lib"] == "numba":
        if runtime_assets is None or not runtime_assets.numba_warmed:
            warmup_numba_started = timing_now()
            warmup_numba()
            print_timing("da3.main.warmup_numba", warmup_numba_started)
            if runtime_assets is not None:
                runtime_assets.numba_warmed = True

    da3_streaming = None
    try:
        construct_da3_streaming_started = timing_now()
        da3_streaming = DA3_Streaming(
            image_dir, save_dir, config, runtime_assets=runtime_assets
        )
        print_timing("da3.main.construct_da3_streaming", construct_da3_streaming_started)
        run_started = timing_now()
        da3_streaming.run()
        print_timing("da3.main.run", run_started)
        close_started = timing_now()
        da3_streaming.close()
        print_timing("da3.main.close", close_started)
    finally:
        if da3_streaming is not None:
            da3_streaming.reset_session_state()
        del da3_streaming
        if runtime_assets is None:
            torch.cuda.empty_cache()
        gc.collect()

    all_ply_path = os.path.join(save_dir, "pcd/combined_pcd.ply")
    input_dir = os.path.join(save_dir, "pcd")
    print("Saving all the point clouds")
    merge_started = timing_now()
    merge_ply_files(input_dir, all_ply_path)
    print_timing("da3.main.merge_ply_files", merge_started)
    print("DA3-Streaming done.")
    total_sec = print_timing("da3.main.total", main_started)
    return {
        "ok": True,
        "save_dir": save_dir,
        "combined_ply": all_ply_path,
        "runtime_cache_key": getattr(runtime_assets, "cache_key", None),
        "total_sec": round(total_sec, 3),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="DA3-Streaming")
    parser.add_argument("--image_dir", type=str, required=True, help="Image path")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="./configs/base_config.yaml",
        help="Image path",
    )
    parser.add_argument("--output_dir", type=str, required=False, default=None, help="Output path")
    args = parser.parse_args(argv)
    run_da3_job(args.image_dir, args.config, args.output_dir)


if __name__ == "__main__":
    main()
    sys.exit()
