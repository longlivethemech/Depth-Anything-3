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
import os
import sys
import time
from pathlib import Path
import faiss
import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SALAD_ROOT = os.path.join(CURRENT_DIR, "salad")
if SALAD_ROOT not in sys.path:
    sys.path.insert(0, SALAD_ROOT)
from loop_utils.salad.models import helper


def timing_now():
    return time.perf_counter()


def print_timing(label, started_at):
    elapsed = time.perf_counter() - started_at
    print(f"[TIMING] {label}: {elapsed:.3f}s")
    return elapsed


class VPRModel(nn.Module):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(
        self,
        # ---- Backbone
        backbone_arch="resnet50",
        backbone_config={},
        # ---- Aggregator
        agg_arch="ConvAP",
        agg_config={},
    ):
        super().__init__()

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config

        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x


class LoopDetector:
    """Loop detector class for detecting loop closures in image sequences"""

    def __init__(self, image_dir, output="loop_closures.txt", config=None):
        """Initialize the loop detector

        Args:
            image_dir: Directory path containing images
            ckpt_path: Model checkpoint path
            image_size: Image resize dimensions [height width]
            batch_size: Batch size for processing
            similarity_threshold: Similarity threshold for loop closure
            top_k: Number of nearest neighbors to check for each image
            use_nms: Whether to use Non-Maximum Suppression (NMS) filtering
            nms_threshold: NMS threshold for minimum frame difference between loop pairs
            output: Output file path
        """
        self.config = config
        self.image_dir = image_dir
        self.ckpt_path = self.config["Weights"]["SALAD"]
        self.image_size = self.config["Loop"]["SALAD"]["image_size"]
        self.batch_size = self.config["Loop"]["SALAD"]["batch_size"]
        self.similarity_threshold = self.config["Loop"]["SALAD"]["similarity_threshold"]
        self.top_k = self.config["Loop"]["SALAD"]["top_k"]
        self.candidate_mode = self.config["Loop"]["SALAD"].get("candidate_mode", "legacy_top_k")
        self.min_pair_frame_gap = int(self.config["Loop"]["SALAD"].get("min_pair_frame_gap", 10))
        self.use_nms = self.config["Loop"]["SALAD"]["use_nms"]
        self.nms_threshold = self.config["Loop"]["SALAD"]["nms_threshold"]
        self.nms_mode = self.config["Loop"]["SALAD"].get("nms_mode", "legacy")
        self.revisit_min_frame_gap = int(
            self.config["Loop"]["SALAD"].get(
                "revisit_min_frame_gap",
                max(90, int(self.nms_threshold) * 3),
            )
        )
        self.revisit_nms_threshold = int(
            self.config["Loop"]["SALAD"].get(
                "revisit_nms_threshold",
                self.nms_threshold,
            )
        )
        self.output = output

        self.model = None
        self.device = None
        self.image_paths = None
        self.descriptors = None
        self.raw_loop_closures = None
        self.loop_closures = None
        self.similarity_matrix = None
        self.nms_stats = {}

    def _input_transform(self, image_size=None):
        """Create image transformation function"""
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        if image_size:
            return T.Compose(
                [
                    T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )
        else:
            return T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

    def load_model(self):
        """Load model"""
        load_model_started = timing_now()
        construct_model_started = timing_now()
        model = VPRModel(
            backbone_arch="dinov2_vitb14",
            backbone_config={
                "num_trainable_blocks": 4,
                "return_token": True,
                "norm_layer": True,
            },
            agg_arch="SALAD",
            agg_config={
                "num_channels": 768,
                "num_clusters": 64,
                "cluster_dim": 128,
                "token_dim": 256,
            },
        )

        print_timing("loop.load_model.construct_model", construct_model_started)
        load_state_dict_started = timing_now()
        model.load_state_dict(torch.load(self.ckpt_path))
        print_timing("loop.load_model.load_state_dict", load_state_dict_started)
        model = model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_to_device_started = timing_now()
        model = model.to(device)
        print_timing("loop.load_model.model_to_device", model_to_device_started)
        print(f"Model loaded: {self.ckpt_path}")

        self.model = model
        self.device = device
        print_timing("loop.load_model.total", load_model_started)
        return model, device

    def get_image_paths(self):
        """Get paths of all image files in directory"""
        get_image_paths_started = timing_now()
        image_extensions = [".jpg", ".jpeg", ".png"]
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(list(Path(self.image_dir).glob(f"*{ext}")))
            image_paths.extend(list(Path(self.image_dir).glob(f"*{ext.upper()}")))

        image_paths = sorted(image_paths)
        self.image_paths = image_paths
        print_timing("loop.get_image_paths.total", get_image_paths_started)
        return image_paths

    def extract_descriptors(self):
        """Extract image feature descriptors"""
        extract_descriptors_started = timing_now()
        if self.model is None or self.device is None:
            self.load_model()

        if self.image_paths is None:
            self.get_image_paths()

        transform = self._input_transform(self.image_size)
        descriptors = []
        batch_prepare_total_sec = 0.0
        batch_forward_total_sec = 0.0

        for i in tqdm(
            range(0, len(self.image_paths), self.batch_size), desc="Extracting features"
        ):
            batch_paths = self.image_paths[i : i + self.batch_size]
            batch_imgs = []

            batch_prepare_started = timing_now()
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    img = transform(img)
                    batch_imgs.append(img)
                except Exception as e:
                    print(f"Error processing image {path}: {e}")
                    img = (
                        torch.zeros(3, 224, 224)
                        if self.image_size is None
                        else torch.zeros(3, self.image_size[0], self.image_size[1])
                    )
                    batch_imgs.append(img)

            batch_prepare_total_sec += time.perf_counter() - batch_prepare_started
            batch_tensor = torch.stack(batch_imgs).to(self.device)

            batch_forward_started = timing_now()
            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16
                ):
                    batch_descriptors = self.model(batch_tensor).cpu()

            batch_forward_total_sec += time.perf_counter() - batch_forward_started
            descriptors.append(batch_descriptors)

        self.descriptors = torch.cat(descriptors)
        print(f"[TIMING] loop.extract_descriptors.batch_prepare_total: {batch_prepare_total_sec:.3f}s")
        print(f"[TIMING] loop.extract_descriptors.batch_forward_total: {batch_forward_total_sec:.3f}s")
        print_timing("loop.extract_descriptors.total", extract_descriptors_started)
        return self.descriptors

    def _apply_endpoint_nms_filter(self, loop_closures, nms_threshold):
        """Apply the original endpoint-suppression NMS used by DA3."""
        if not loop_closures or nms_threshold <= 0:
            return loop_closures

        sorted_loops = sorted(loop_closures, key=lambda x: x[2], reverse=True)
        filtered_loops = []
        suppressed = set()

        max_frame = max(max(idx1, idx2) for idx1, idx2, _ in loop_closures)

        for idx1, idx2, sim in sorted_loops:
            if idx1 in suppressed or idx2 in suppressed:
                continue

            filtered_loops.append((idx1, idx2, sim))

            suppress_range = set()

            start1 = max(0, idx1 - nms_threshold)
            end1 = min(idx1 + nms_threshold + 1, idx2)
            suppress_range.update(range(start1, end1))

            start2 = max(idx1 + 1, idx2 - nms_threshold)
            end2 = min(idx2 + nms_threshold + 1, max_frame + 1)
            suppress_range.update(range(start2, end2))

            suppressed.update(suppress_range)

        return filtered_loops

    def _apply_pair_window_nms_filter(self, loop_closures, nms_threshold):
        """Suppress only candidates that describe the same two loop windows.

        Endpoint NMS is too aggressive for online SLAM because a high-scoring
        local match near one endpoint can suppress a true long-baseline revisit.
        For revisit candidates we only suppress a new pair when both endpoints
        are close to an already kept pair.
        """
        if not loop_closures or nms_threshold <= 0:
            return loop_closures

        sorted_loops = sorted(loop_closures, key=lambda x: x[2], reverse=True)
        filtered_loops = []

        for idx1, idx2, sim in sorted_loops:
            same_loop_window = False
            for kept_idx1, kept_idx2, _ in filtered_loops:
                if (
                    abs(idx1 - kept_idx1) <= nms_threshold
                    and abs(idx2 - kept_idx2) <= nms_threshold
                ):
                    same_loop_window = True
                    break
            if same_loop_window:
                continue
            filtered_loops.append((idx1, idx2, sim))

        return filtered_loops

    def _apply_nms_filter(self, loop_closures, nms_threshold):
        """Apply Non-Maximum Suppression (NMS) filtering to loop pairs"""
        if not loop_closures or nms_threshold <= 0:
            self.nms_stats = {
                "mode": self.nms_mode,
                "raw_count": len(loop_closures or []),
                "filtered_count": len(loop_closures or []),
                "revisit_min_frame_gap": int(self.revisit_min_frame_gap),
            }
            return loop_closures

        if self.nms_mode != "gap_aware":
            filtered = self._apply_endpoint_nms_filter(loop_closures, nms_threshold)
            self.nms_stats = {
                "mode": "legacy",
                "raw_count": len(loop_closures),
                "filtered_count": len(filtered),
                "nms_threshold": int(nms_threshold),
                "revisit_min_frame_gap": int(self.revisit_min_frame_gap),
            }
            return filtered

        revisit_min_frame_gap = max(int(self.revisit_min_frame_gap), 0)
        revisit_nms_threshold = max(int(self.revisit_nms_threshold), 0)
        revisit_loops = []
        local_loops = []
        for idx1, idx2, sim in loop_closures:
            if abs(int(idx2) - int(idx1)) >= revisit_min_frame_gap:
                revisit_loops.append((idx1, idx2, sim))
            else:
                local_loops.append((idx1, idx2, sim))

        revisit_filtered = self._apply_pair_window_nms_filter(
            revisit_loops,
            revisit_nms_threshold,
        )
        local_filtered = self._apply_endpoint_nms_filter(local_loops, nms_threshold)

        # Put true revisits first so any downstream max-new-window cap spends
        # budget on loop closures instead of local repeated views.
        filtered = revisit_filtered + local_filtered
        self.nms_stats = {
            "mode": "gap_aware",
            "raw_count": len(loop_closures),
            "raw_revisit_count": len(revisit_loops),
            "raw_local_count": len(local_loops),
            "filtered_count": len(filtered),
            "filtered_revisit_count": len(revisit_filtered),
            "filtered_local_count": len(local_filtered),
            "nms_threshold": int(nms_threshold),
            "revisit_nms_threshold": int(revisit_nms_threshold),
            "revisit_min_frame_gap": int(revisit_min_frame_gap),
        }
        return filtered

    def _ensure_decending_order(self, tuples_list):
        return [(max(a, b), min(a, b), score) for a, b, score in tuples_list]

    def find_loop_closures(self):
        """Find loop closures"""
        find_loop_closures_started = timing_now()
        if self.descriptors is None:
            self.extract_descriptors()

        embed_size = self.descriptors.shape[1]
        normalized_descriptors = self.descriptors.numpy()
        if str(self.candidate_mode) == "full_pair_ranking":
            similarity_started = timing_now()
            similarity_matrix = normalized_descriptors @ normalized_descriptors.T
            self.similarity_matrix = similarity_matrix
            print_timing("loop.find_loop_closures.full_pair_similarity", similarity_started)

            loop_closures = []
            min_gap = max(int(self.min_pair_frame_gap), 0)
            frame_count = int(similarity_matrix.shape[0])
            for i in range(frame_count):
                for j in range(i + min_gap + 1, frame_count):
                    loop_closures.append((i, j, float(similarity_matrix[i, j])))

            loop_closures.sort(key=lambda x: x[2], reverse=True)
            self.raw_loop_closures = list(loop_closures)
            self.nms_stats = {
                "mode": "full_pair_ranking",
                "raw_count": len(loop_closures),
                "filtered_count": len(loop_closures),
                "min_pair_frame_gap": int(min_gap),
                "similarity_threshold_used": False,
                "top_k_used": False,
            }
            self.loop_closures = self._ensure_decending_order(loop_closures)
            print_timing("loop.find_loop_closures.total", find_loop_closures_started)
            return self.loop_closures

        faiss_index_started = timing_now()
        faiss_index = faiss.IndexFlatIP(embed_size)
        faiss_index.add(normalized_descriptors)
        print_timing("loop.find_loop_closures.faiss_index_add", faiss_index_started)

        faiss_search_started = timing_now()
        similarities, indices = faiss_index.search(
            normalized_descriptors, self.top_k + 1
        )  # +1 because self is most similar
        print_timing("loop.find_loop_closures.faiss_search", faiss_search_started)

        loop_closures = []
        for i in range(len(self.descriptors)):
            # Skip first result (self)
            for j in range(1, self.top_k + 1):
                neighbor_idx = indices[i, j]
                similarity = similarities[i, j]

                if similarity > self.similarity_threshold and abs(i - neighbor_idx) > self.min_pair_frame_gap:
                    if i < neighbor_idx:
                        loop_closures.append((i, neighbor_idx, similarity))
                    else:
                        loop_closures.append((neighbor_idx, i, similarity))

        loop_closures = list(set(loop_closures))
        loop_closures.sort(key=lambda x: x[2], reverse=True)
        self.raw_loop_closures = list(loop_closures)

        if self.use_nms and self.nms_threshold > 0:
            loop_closures = self._apply_nms_filter(loop_closures, self.nms_threshold)
        else:
            self.nms_stats = {
                "mode": "disabled",
                "raw_count": len(loop_closures),
                "filtered_count": len(loop_closures),
                "revisit_min_frame_gap": int(self.revisit_min_frame_gap),
            }

        self.loop_closures = self._ensure_decending_order(loop_closures)
        print_timing("loop.find_loop_closures.total", find_loop_closures_started)
        return self.loop_closures

    def save_results(self):
        """Save loop detection results to file"""
        save_results_started = timing_now()
        if self.loop_closures is None:
            self.find_loop_closures()

        with open(self.output, "w") as f:
            f.write("# Loop Detection Results (index1, index2, similarity)\n")
            if self.use_nms:
                f.write(f"# NMS filtering applied, threshold: {self.nms_threshold}\n")
                for key, value in sorted(self.nms_stats.items()):
                    f.write(f"# NMS {key}: {value}\n")
            f.write("\n# Loop pairs:\n")
            for i, j, sim in self.loop_closures:
                f.write(f"{i}, {j}, {sim:.4f}\n")
            f.write("\n# Image path list:\n")
            for i, path in enumerate(self.image_paths):
                f.write(f"# {i}: {path}\n")

        print(f"Found {len(self.loop_closures)} loop pairs, results saved to {self.output}")
        if self.use_nms:
            print(f"NMS filtering applied, threshold: {self.nms_threshold}")

        if self.loop_closures:
            print("\nTop 10 loop pairs:")
            for i, (idx1, idx2, sim) in enumerate(self.loop_closures[:10]):
                print(f"{idx1}, {idx2}, similarity: {sim:.4f}")
                if i >= 9:
                    break

        print_timing("loop.save_results.total", save_results_started)

    def get_loop_list(self):
        return [(idx1, idx2) for idx1, idx2, _ in self.loop_closures]

    def run(self):
        """Run complete loop detection pipeline"""
        run_started = timing_now()
        print("Loading model...")
        if self.model is None:
            load_model_started = timing_now()
            self.load_model()
            print_timing("loop.run.load_model_if_needed", load_model_started)

        get_image_paths_started = timing_now()
        self.get_image_paths()
        print_timing("loop.run.get_image_paths", get_image_paths_started)
        if not self.image_paths:
            print(f"No image files found in {self.image_dir}")
            return

        print(f"Found {len(self.image_paths)} image files")

        extract_descriptors_started = timing_now()
        self.extract_descriptors()
        print_timing("loop.run.extract_descriptors", extract_descriptors_started)

        find_loop_closures_started = timing_now()
        self.find_loop_closures()
        print_timing("loop.run.find_loop_closures", find_loop_closures_started)

        save_results_started = timing_now()
        self.save_results()
        print_timing("loop.run.save_results", save_results_started)

        print_timing("loop.run.total", run_started)
        return self.loop_closures


def main():
    parser = argparse.ArgumentParser(description="Loop detection using SALAD model")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/media/deng/Data/KITTIdataset/data_odometry_color/dataset/sequences/00/image_2",
        help="Directory path containing images",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="./weights/dino_salad.ckpt", help="Model checkpoint path"
    )
    parser.add_argument(
        "--image_size",
        nargs=2,
        type=int,
        default=[336, 336],
        help="Image resize dimensions [height width]",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for loop closure",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of nearest neighbors to check for each image"
    )
    parser.add_argument("--output", type=str, default="loop_closures.txt", help="Output file path")
    parser.add_argument(
        "--use_nms",
        action="store_true",
        default=True,
        help="Whether to use Non-Maximum Suppression (NMS) filtering",
    )
    parser.add_argument(
        "--nms_threshold",
        type=int,
        default=25,
        help="NMS threshold for minimum frame difference between loop pairs",
    )

    args = parser.parse_args()

    detector = LoopDetector(
        image_dir=args.image_dir,
        ckpt_path=args.ckpt_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        similarity_threshold=args.similarity_threshold,
        top_k=args.top_k,
        use_nms=args.use_nms,
        nms_threshold=args.nms_threshold,
        output=args.output,
    )

    detector.run()


if __name__ == "__main__":
    main()
