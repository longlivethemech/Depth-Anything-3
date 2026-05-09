# AutoDL Runtime Patches

Last updated: 2026-05-08

This branch preserves the Depth-Anything-3 runtime that is currently used by the Streaming Replay MVP on AutoDL.

Current handoff branch / HEAD:

```text
branch: autodl-da3-runtime-patches-20260507
HEAD: 3833494d1d588d39ffccc5c6737c9be76fe9ab00
```

## Source

The patch set was copied from the AutoDL shared canonical repository:

```text
/autodl-fs/data/code/Depth-Anything-3
```

The 526 runtime copy was also checked and matched the shared canonical tree:

```text
/root/autodl-tmp/code/Depth-Anything-3
```

Both trees were based on upstream `ByteDance-Seed/Depth-Anything-3` at:

```text
41736238f5bced4debf3f2a12375d2466874866d
```

## Included Changes From 2026-05-07 Rescue Commit

- `da3_streaming/da3_streaming.py`
  - Adds a single-chunk path that writes `pcd/0_pcd.ply`, saves depth/conf outputs when enabled, writes camera poses, and returns before the multi-chunk merge path.
  - Fixes camera pose saving for single-chunk inputs by not dropping the tail overlap when there is no following chunk.
- `da3_streaming/configs/base_config_auto1m.yaml`
  - Preserves the high-density AutoDL config observed in shared canonical.
- `scripts/da3_multiview_html.py`
  - Preserves the quickflow/static multiview helper observed in shared canonical.
- `.bootstrap_autodl_da3_multiview.*` and `da3_streaming/.bootstrap_autodl_da3.*`
  - Preserves the bootstrap state/env files observed in shared canonical. These files are runtime documentation, not portable secrets.
- `da3_streaming/loop_utils/salad`
  - Points to `longlivethemech/salad` branch `autodl-da3-runtime-patches-20260507`.
  - That salad fork commit adds a local torch hub fallback for DINOv2 so AutoDL does not randomly depend on GitHub network access when `/root/.cache/torch/hub/facebookresearch_dinov2_main` exists.

## Included Changes From 2026-05-08 Runtime Commit

Commit:

```text
7d55f32c3bc7e2f3060bba6ba65c439ae01d9e55
da3_streaming: add runtime timings and parallel save-depth export
```

Changed files:

```text
da3_streaming/da3_streaming.py
da3_streaming/loop_utils/loop_detector.py
```

Runtime behavior:

- Adds timing instrumentation for DA3 streaming runtime phases.
- Splits timing around `init / inference / loop / alignment / export / save_depth_conf_result`.
- Changes `save_depth_conf_result()` from serial `np.savez_compressed` calls to parallel `ThreadPoolExecutor` export.
- Keeps the downstream contract stable:
  - `.npz` format is unchanged.
  - `frame_{idx}.npz` naming is unchanged.
  - Existing downstream readers should not need a format change.
- Makes the default save-depth worker count cgroup-aware:
  - reads `/sys/fs/cgroup/cpu.max` first when available.
  - avoids trusting misleading `os.cpu_count()` / `sched_getaffinity()` values on AutoDL containers.
  - 504 was observed as 25 effective vCPU, with default `max_workers=24`.

## Included Changes From 2026-05-08 Resident Runtime Assets Commit

Commit:

```text
3833494d1d588d39ffccc5c6737c9be76fe9ab00
da3_streaming: add explicit runtime-assets resident API
```

Changed files:

```text
da3_streaming/da3_streaming.py
```

Runtime behavior:

- Adds explicit resident entry points:
  - `build_runtime_assets()`
  - `runtime_assets_cache_key()`
  - `run_da3_job(..., runtime_assets=...)`
- Allows `streaming-replay-mvp/da3_resident_runner.py` to reuse loaded DA3 weights and loop-detector assets across chunk jobs.
- Preserves the CLI path while giving the relay worker a stable API for cold-to-warm chunk execution.
- Observed on AutoDL 526 during `warm526-20260509-1714`:
  - chunk 0 cold wall time: `25.029s`
  - chunk 1 warm wall time: `5.874s`
  - warm `runner_prepare_context_sec`: `0.004s`
  - warm `runner_api_run_sec`: `5.863s`
  - warm runtime cache hit: `true`

## AutoDL 504 Handoff Check

For developers connecting to AutoDL 504, verify the canonical shared repo first:

```bash
cd /autodl-fs/data/code/Depth-Anything-3
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
```

Then verify the local runtime mirror:

```bash
cd /root/autodl-tmp/code/Depth-Anything-3
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
```

Both paths are expected to report:

```text
branch = autodl-da3-runtime-patches-20260507
HEAD   = 3833494d1d588d39ffccc5c6737c9be76fe9ab00
origin = https://github.com/longlivethemech/Depth-Anything-3.git
```

The current resident validation job repo may exist at:

```text
/root/autodl-tmp/jobs/20260508-1754-da3-resident-phase1-review/Depth-Anything-3
```

That path is an experiment scene, not the canonical source. Do not use it to overwrite shared canonical or this organization fork.

## Not Included

- `da3_streaming/weights`
  - The AutoDL tree contains an absolute symlink to shared weights. That is deployment state and should be created by AutoDL init/runbook steps, not committed into this repository.
- `*.bak` files
  - The remote tree contains historical backup files for `da3_streaming.py` and salad `dinov2.py`. The live patch is now captured by git instead.
- `da3_streaming/loop_utils/salad` as an untracked plain directory
  - The committed repo uses the `salad` submodule pointer. Some AutoDL runtime trees may still show an untracked `da3_streaming/loop_utils/salad` hygiene entry. Treat it as deployment noise unless the submodule pointer itself changes.

## Deployment Rule

Treat this repository branch as the source for the AutoDL DA3 runtime. Do not replace `/autodl-fs/data/code/Depth-Anything-3` with a clean upstream checkout unless these runtime patches have been intentionally removed and the Streaming Replay MVP has been retested with `--enable-real-da3`.

Before syncing this repo back to AutoDL shared canonical, run a dry-run first and verify that `da3_streaming/weights` will be restored as a symlink to the shared weights directory after sync.
