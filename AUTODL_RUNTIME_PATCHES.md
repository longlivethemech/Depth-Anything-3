# AutoDL Runtime Patches

Date: 2026-05-07

This branch preserves the Depth-Anything-3 runtime that is currently used by the Streaming Replay MVP on AutoDL.

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

## Included Changes

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

## Not Included

- `da3_streaming/weights`
  - The AutoDL tree contains an absolute symlink to shared weights. That is deployment state and should be created by AutoDL init/runbook steps, not committed into this repository.
- `*.bak` files
  - The remote tree contains historical backup files for `da3_streaming.py` and salad `dinov2.py`. The live patch is now captured by git instead.

## Deployment Rule

Treat this repository branch as the source for the AutoDL DA3 runtime. Do not replace `/autodl-fs/data/code/Depth-Anything-3` with a clean upstream checkout unless these runtime patches have been intentionally removed and the Streaming Replay MVP has been retested with `--enable-real-da3`.

Before syncing this repo back to AutoDL shared canonical, run a dry-run first and verify that `da3_streaming/weights` will be restored as a symlink to the shared weights directory after sync.
