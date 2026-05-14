# DA3 Confidence-Aware Fusion Handoff

Date: 2026-05-12

This handoff covers the reasoning, current experimental code state, and next execution tasks for improving DA3-Streaming point-cloud output. The goal is to make repeated observations increase geometric confidence instead of producing duplicate ghost surfaces, while also reducing holes where video clearly sees a region but the point cloud drops it.

## Workspace

Use the isolated copy, not the original DA3 checkout:

```text
/Users/mechforge1/projects/github-organization-staging-round3/Depth-Anything-3-confidence-fusion
```

Current branch in that copy:

```text
codex/da3-confidence-fusion-20260510
```

The original `Depth-Anything-3` directory was intentionally left separate.

## Problem Summary

The current DA3-Streaming output often shows two opposite symptoms:

1. Ghosting: the same object appears as two or more offset point layers.
2. Holes: RGB video clearly contains a region, but the point-cloud viewer has gaps.

These are related but not identical.

Ghosting usually comes from chunk-level point clouds being appended after imperfect Sim3 alignment. Repeated observations are not fused into a single map estimate. Holes usually come from hard confidence filtering: if a region has low DA3 confidence in one chunk, those pixels are dropped before they can accumulate evidence from later views.

## Original DA3-Streaming Behavior

The original pipeline is not a probabilistic fusion pipeline. It is closer to:

```text
split video into overlapping chunks
run DA3 independently per chunk
estimate Sim3 between neighboring chunks using overlap
apply accumulated Sim3 to each chunk
filter points by confidence
write each chunk as PLY
concatenate PLY files
```

Important implementation points:

- `get_chunk_indices()` uses `chunk_size` and `overlap`.
- `process_single_chunk()` runs `self.model.inference(...)` and saves `chunk_N.npy`.
- Adjacent chunks are aligned using overlap frames:
  - previous chunk last `overlap` frames
  - current chunk first `overlap` frames
  - `align_2pcds()` -> `weighted_align_point_maps()`
- Export uses a hard threshold:

```python
conf_threshold = np.mean(confs) * Pointcloud_Save.conf_threshold_coef
```

- `merge_ply_files()` concatenates PLY payloads. It does not merge duplicate surfaces.

Consequence:

```text
High-confidence repeat observations become duplicate layers.
Low-confidence repeat observations are often discarded before they can help.
```

## Why Holes Happen Even When Video Sees the Region

RGB visibility is not enough for point-cloud visibility. A pixel must also have:

- a valid depth estimate
- usable confidence
- a stable pose/depth relationship
- no later filtering or conflict rejection

Common hole sources:

- dark/black surfaces
- specular highlights and reflective areas
- motion blur
- H264 compression noise
- thin structures and depth discontinuities
- grazing-angle surfaces
- repeated texture
- high per-chunk confidence mean causing hard threshold to drop difficult local regions

The important design lesson: low confidence should not always mean immediate deletion. It should often mean low-weight evidence that needs repeated support before being exported.

## Existing Experimental MVP

New file:

```text
da3_streaming/loop_utils/map_fusion.py
```

New class:

```python
VoxelSurfelFusion
```

The MVP is pure NumPy and has no new dependency.

Current behavior:

- Maintains a voxel hash of surfels.
- Repeated observations within `fusion_radius` update one surfel:
  - weighted position
  - weighted color
  - accumulated weight
  - observation count
  - variance
- Observations near an existing surfel but outside `fusion_radius` and within `conflict_radius` are treated as conflicts and skipped.
- Far observations create new surfels.
- Export is still standard binary PLY point cloud.

DA3 streaming integration:

- `DA3_Streaming.__init__` now initializes fusion state when config enables it.
- `export_streaming_chunk_artifacts()` integrates the chunk into the fusion map and exports the current fused map.
- `export_streaming_corrected_chunk_artifacts()` rebuilds the fused map from corrected transforms and returns one corrected fused-map artifact.
- Returned artifacts include:

```text
pointcloud_mode: confidence_voxel_surfel_fused_map
requires_map_replace: True
fusion_stats: {...}
```

Configs now include:

```yaml
Pointcloud_Fusion:
  enabled: True
  voxel_size: 0.02
  fusion_radius: 0.03
  conflict_radius: 0.06
  integrate_sample_ratio: 1.0
  min_weight: 0.0001
  max_weight: 100.0
  min_export_weight: 0.0
  min_export_observations: 1
  max_export_points: 0
  seed: 42
```

Config files touched:

- `da3_streaming/configs/base_config.yaml`
- `da3_streaming/configs/base_config_auto1m.yaml`
- `da3_streaming/configs/kitti.yaml`
- `da3_streaming/configs/tum.yaml`

## Verification Status

Per user instruction, do not run DA3, Python compilation, or tests in this branch yet. The active runtime route is occupied by another code branch.

Only static checks were done:

- `git diff --check`
- trailing whitespace scan
- manual code inspection of the changed paths

No runtime behavior has been verified.

## Important Viewer/Bridge Warning

When fusion is enabled, each exported PLY is a full fused map state, not a raw per-chunk delta.

That means downstream code must not append every fused-map artifact to the viewer. It must replace the current map.

The next project-owning Codex session should adapt `streaming-replay-mvp`:

- detect `requires_map_replace: True`, or
- detect `pointcloud_mode == "confidence_voxel_surfel_fused_map"`

Then:

```text
send map_reset
load the new fused map geometry
do not append it as another chunk layer
```

Without this bridge/viewer adaptation, the browser may still show duplicate maps even though DA3 is exporting fused maps.

## Better MVP Direction

The first fusion MVP helps with ghosting, but holes need a slightly better confidence policy.

Do not simply lower the global confidence threshold a lot. That will bring back noisy depth, edge fuzz, and ghosting.

Recommended next MVP:

```text
low threshold at integrate time
higher confirmation threshold at export time
```

In other words:

```text
very bad confidence -> discard
medium/low confidence -> candidate surfel
high confidence or repeated consistent observations -> confirmed surfel
confirmed surfels -> exported PLY
```

Suggested config additions:

```yaml
Pointcloud_Fusion:
  integrate_conf_threshold_coef: 0.15
  confirm_observations: 2
  confirm_weight: 1.5
  instant_confirm_weight: 3.0
  candidate_conflict_policy: skip_against_confirmed
```

Suggested logic:

1. Integrate threshold should be much lower than old point-cloud export threshold.
2. Confidence should mostly become a weight, not a binary keep/drop decision.
3. Low-confidence surfels should be retained as candidates.
4. Export should include:
   - high-weight single observations, or
   - repeated consistent low/medium-confidence observations.
5. Conflict behavior should distinguish confirmed and candidate surfels:
   - confirmed surfel nearby: skip conflicting observation
   - only candidate nearby: allow another candidate or update based on consistency

This is a small extension and should be done before heavier TSDF, mesh, or neural fusion work.

## Suggested Execution Tasks

1. Inspect current experimental changes.

```text
git status --short --branch
git diff -- da3_streaming/da3_streaming.py
```

Do not run DA3 until the user confirms the runtime route is free.

2. Extend `VoxelSurfelFusion` with candidate/confirmed surfel state.

Fields to add per surfel:

```text
state: candidate | confirmed
first_seen_integration
last_seen_integration
low_conf_observations
confirmed_reason
```

3. Add soft integration threshold.

Do not call `_pointcloud_conf_threshold()` directly for fusion integration. Add a separate threshold:

```python
integrate_threshold = mean(conf) * integrate_conf_threshold_coef
```

4. Add export gating.

Export only if:

```text
surfel.weight >= instant_confirm_weight
or surfel.observations >= confirm_observations and surfel.weight >= confirm_weight
```

5. Add debug stats.

Return these in `fusion_stats`:

```text
candidate_count
confirmed_count
low_conf_integrated
low_conf_confirmed
conflict_count
exported_point_count
```

6. Update `streaming-replay-mvp` bridge after DA3-side change.

The bridge must map fused-map artifacts to map replacement, not append.

7. Only after code review, run a controlled comparison:

```text
fusion off
fusion on, current MVP
fusion on, candidate/confirmed MVP
```

Measure:

- exported point count
- visible hole reduction
- visible ghost reduction
- conflict count
- repeated-observation confirmation rate

## Letter To The Next Codex Session

Dear next Codex session,

You are taking over an isolated DA3 experiment at:

```text
/Users/mechforge1/projects/github-organization-staging-round3/Depth-Anything-3-confidence-fusion
```

The user's goal is not merely to reduce duplicate point clouds. The real goal is better mapping semantics:

```text
When the same region is observed multiple times, the map should become more confident and more accurate, not create repeated ghost layers.
When a region is visible in video but has low single-frame confidence, it should have a path to become visible after repeated consistent observations.
```

I implemented a first confidence-aware voxel/surfel fusion MVP. It adds `VoxelSurfelFusion` and wires it into DA3-Streaming exports. It currently fuses repeated close observations, skips near conflicts, and exports a standard PLY fused map. Artifacts now declare `pointcloud_mode=confidence_voxel_surfel_fused_map` and `requires_map_replace=True`.

Please do not assume the downstream viewer is already compatible. The current viewer/bridge likely still appends `chunk_geometry`. For fused-map artifacts, that is wrong. The bridge should send a `map_reset` and replace the geometry whenever `requires_map_replace=True`.

The next best improvement is not TSDF or mesh reconstruction yet. Keep it MVP-scale:

1. Add candidate versus confirmed surfel states.
2. Lower the integration confidence threshold.
3. Use confidence as weight, not a hard gate.
4. Export only surfels that have high confidence or repeated consistent support.
5. Keep conflict handling conservative around confirmed surfels.
6. Add debug stats so the user can see whether holes are caused by no observations, low-confidence candidates, or conflict rejection.

The DA3 runtime route is now available. It is OK to run DA3, run tests, and do controlled comparisons for this experiment.

Be careful with the original repo. Work in the copied folder above unless the user explicitly asks otherwise.

Sincerely,

The previous Codex session
