#!/usr/bin/env python3
"""DA3 多图重建 → 双轨导出（dense source point cloud + lighter preview HTML）

通用重建脚本：读取一个图片目录中的多张静态图片，调用 DA3 / DA3Nested 做多视角推理，
再把预测的 depth + intrinsics + extrinsics 反投影成彩色点云。

默认输出一份浏览器友好的 preview HTML；可选同时导出一份更密的 source point cloud
（NPZ / PLY），供后续 support map / adaptive mesh / mesh 后处理使用。
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DA3 多图重建 → 双轨导出")
    p.add_argument("--input-dir", required=True, help="输入图片目录")
    p.add_argument("--output-html", required=True, help="输出 preview HTML 路径")
    p.add_argument("--glob", default="*.jpg", help="图片匹配模式，默认 *.jpg")
    p.add_argument("--model-dir", default="/root/autodl-fs/da3-weights", help="本地模型目录或 HuggingFace repo id")
    p.add_argument("--process-res", type=int, default=504, help="DA3 process_res，默认 504")
    p.add_argument("--process-res-method", default="upper_bound_resize", help="DA3 resize 策略：upper_bound_resize|lower_bound_resize")
    p.add_argument("--max-points-per-image", type=int, default=8000, help="preview HTML 每张图最多保留多少点，默认 8000")
    p.add_argument("--source-max-points-per-image", type=int, default=0, help="source 点云每张图最多保留多少点；0=保留全部有效点")
    p.add_argument("--output-source-npz", help="可选：输出 dense source point cloud 的 .npz 路径")
    p.add_argument("--output-source-ply", help="可选：输出 dense source point cloud 的 .ply 路径")
    p.add_argument("--output-stats", help="可选：输出统计 JSON 路径")
    p.add_argument("--device", default="auto", help="cuda|mps|cpu|auto")
    p.add_argument("--title", default="DA3 Multiview Point Cloud", help="HTML 标题")
    p.add_argument("--point-size", type=float, default=0.002, help="Three.js 点大小，默认 0.002")
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def sample_points(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points > 0 and len(points) > max_points:
        sel = np.random.choice(len(points), max_points, replace=False)
        return points[sel], colors[sel]
    return points.copy(), colors.copy()


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
    with path.open("wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        data = np.empty(
            len(points),
            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")],
        )
        data["x"] = points[:, 0].astype(np.float32)
        data["y"] = points[:, 1].astype(np.float32)
        data["z"] = points[:, 2].astype(np.float32)
        data["r"] = cols[:, 0]
        data["g"] = cols[:, 1]
        data["b"] = cols[:, 2]
        data.tofile(f)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_html = Path(args.output_html)
    output_source_npz = Path(args.output_source_npz) if args.output_source_npz else None
    output_source_ply = Path(args.output_source_ply) if args.output_source_ply else None
    output_stats = Path(args.output_stats) if args.output_stats else None

    image_paths = sorted(str(p) for p in input_dir.glob(args.glob))
    if not image_paths:
        raise SystemExit(f"ERROR: no images matched {args.glob} in {input_dir}")

    print("=" * 60)
    print(f"DA3 multiview reconstruct: {len(image_paths)} images")
    print(f"  input_dir   = {input_dir}")
    print(f"  output_html = {output_html}")
    if output_source_npz:
        print(f"  source_npz  = {output_source_npz}")
    if output_source_ply:
        print(f"  source_ply  = {output_source_ply}")
    print(f"  model_dir   = {args.model_dir}")
    print("=" * 60)

    print("\n[1] 读取图片...")
    images = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            raise SystemExit(f"ERROR: failed to read {p}")
        images.append(img)
        print(f"    {Path(p).name}: {img.shape[1]}x{img.shape[0]}")

    print("\n[2] 加载模型...")
    t0 = time.time()
    device = choose_device(args.device)
    print(f"    device = {device}")
    model = DepthAnything3.from_pretrained(args.model_dir)
    model = model.to(device=device)
    print(f"    load_time = {time.time() - t0:.1f}s")

    print(f"\n[3] 多视角推理 (process_res={args.process_res})...")
    t1 = time.time()
    prediction = model.inference(image_paths, process_res=args.process_res, process_res_method=args.process_res_method)
    t_infer = time.time() - t1
    print(f"    infer_time = {t_infer:.1f}s")

    depths = prediction.depth
    extrs = prediction.extrinsics
    intrs = prediction.intrinsics
    is_metric = prediction.is_metric
    proc_imgs = prediction.processed_images

    print("\n[4] 结果摘要:")
    print(f"    depth      = {depths.shape}")
    print(f"    extrinsics = {None if extrs is None else extrs.shape}")
    print(f"    intrinsics = {None if intrs is None else intrs.shape}")
    print(f"    is_metric  = {is_metric}")

    if extrs is None or intrs is None:
        raise SystemExit("ERROR: prediction missing extrinsics/intrinsics; cannot build multiview point cloud")

    print("\n[5] 反投影点云...")
    preview_points_all = []
    preview_colors_all = []
    source_points_all = []
    source_colors_all = []
    cam_positions = []
    cam_rotations_c2w = []
    per_image_stats = []

    export_source = output_source_npz is not None or output_source_ply is not None

    for idx, p in enumerate(image_paths):
        depth = depths[idx]
        H, W = depth.shape
        K = intrs[idx]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        ext = extrs[idx]
        R_c2w = ext[:3, :3].T
        t_c2w = -R_c2w @ ext[:3, 3]

        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        uu = uu.reshape(-1).astype(np.float64)
        vv = vv.reshape(-1).astype(np.float64)
        z = depth[vv.astype(int), uu.astype(int)].astype(np.float64)
        mask = z > 1e-3
        uu, vv, z = uu[mask], vv[mask], z[mask]

        Xc = (uu - cx) / fx * z
        Yc = (vv - cy) / fy * z
        pts_cam = np.stack([Xc, Yc, z], axis=1)
        pts_world = (R_c2w @ pts_cam.T).T + t_c2w

        if proc_imgs is not None:
            cols = proc_imgs[idx][vv.astype(int), uu.astype(int)] / 255.0
        else:
            img_r = cv2.resize(images[idx], (W, H))
            cols = img_r[vv.astype(int), uu.astype(int)][:, ::-1] / 255.0

        full_count = int(len(pts_world))
        preview_pts, preview_cols = sample_points(pts_world, cols, args.max_points_per_image)
        preview_points_all.append(preview_pts)
        preview_colors_all.append(preview_cols)

        source_count = 0
        if export_source:
            source_pts, source_cols = sample_points(pts_world, cols, args.source_max_points_per_image)
            source_points_all.append(source_pts)
            source_colors_all.append(source_cols)
            source_count = int(len(source_pts))

        cam_pos = -ext[:3, :3].T @ ext[:3, 3]
        cam_positions.append(cam_pos.astype(np.float32))
        cam_rotations_c2w.append(R_c2w.astype(np.float32))
        per_image_stats.append(
            {
                "image": Path(p).name,
                "full_points": full_count,
                "preview_points": int(len(preview_pts)),
                "source_points": source_count,
                "camera_position": [float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])],
                "camera_rotation_c2w": R_c2w.astype(np.float32).tolist(),
            }
        )
        print(
            f"    {Path(p).name}: full={full_count:,} | preview={len(preview_pts):,}"
            + (f" | source={source_count:,}" if export_source else "")
            + f" | cam=({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f})"
        )

    preview_points_all = np.concatenate(preview_points_all, axis=0)
    preview_colors_all = np.concatenate(preview_colors_all, axis=0)
    print(f"\n[6] Preview 合并总点数: {len(preview_points_all):,}")

    source_points = None
    source_colors = None
    if export_source:
        source_points = np.concatenate(source_points_all, axis=0)
        source_colors = np.concatenate(source_colors_all, axis=0)
        print(f"    Source 合并总点数: {len(source_points):,}")

    print("\n[7] 生成 preview HTML...")
    preview_centroid = preview_points_all.mean(axis=0)
    pts_c = (preview_points_all - preview_centroid).astype(np.float32)
    cols_f = preview_colors_all.astype(np.float32)
    pos_b64 = base64.b64encode(pts_c.tobytes()).decode()
    col_b64 = base64.b64encode(cols_f.tobytes()).decode()

    colors_hex = [
        0xff4444, 0xff8844, 0xffcc44, 0x44ff44, 0x44ffcc,
        0x4488ff, 0x8844ff, 0xff44cc, 0xffffff, 0x00ffff,
    ]
    camera_pose_records = []
    for idx, p in enumerate(image_paths):
        cp = (cam_positions[idx] - preview_centroid).astype(np.float32)
        camera_pose_records.append(
            {
                "image": Path(p).name,
                "position": [float(cp[0]), float(cp[1]), float(cp[2])],
                "rotation_c2w": cam_rotations_c2w[idx].astype(np.float32).tolist(),
                "color": int(colors_hex[idx % len(colors_hex)]),
            }
        )
    camera_pose_json = json.dumps(camera_pose_records, ensure_ascii=False)

    depth_unit = "metric(m)" if is_metric else "relative"
    info_lines = [
        args.title,
        f"Images: {len(image_paths)} | Preview points: {len(pts_c):,} | Depth: {depth_unit}",
        f"Infer: {t_infer:.1f}s | process_res={args.process_res} | process_res_method={args.process_res_method} | preview_max_per_image={args.max_points_per_image}",
    ]
    if export_source:
        info_lines.append(
            f"Source points: {len(source_points):,} | source_max_per_image={'all' if args.source_max_points_per_image == 0 else args.source_max_points_per_image}"
        )
    info_lines.append("Colored spheres = camera positions | RGB lines = camera axes (X/Y/Z)")
    info = "<br>".join(info_lines)

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>{args.title}</title>
<style>body{{margin:0;overflow:hidden;background:#0f111a}} #info{{position:absolute;top:10px;left:10px;color:#eaeaea;font:13px monospace;background:rgba(0,0,0,.6);padding:8px 12px;border-radius:6px;z-index:10;line-height:1.6}}</style></head><body>
<div id='info'>{info}</div>
<script type='importmap'>{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}}}</script>
<script type='module'>
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
function b64f(b){{const a=atob(b),u=new Uint8Array(a.length);for(let i=0;i<a.length;i++)u[i]=a.charCodeAt(i);return new Float32Array(u.buffer)}}
const P=b64f('{pos_b64}'),C=b64f('{col_b64}');
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0f111a);
const camera=new THREE.PerspectiveCamera(60,innerWidth/innerHeight,0.001,100);
// Keep Three.js default Y-up so raw axes stay X=red, Y=green, Z=blue.
// For DA3 room previews, floor interpretation should be checked against the raw Y axis
// instead of force-remapping the viewer to Z-up here.
const renderer=new THREE.WebGLRenderer({{antialias:true}});renderer.setSize(innerWidth,innerHeight);renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);
const geom=new THREE.BufferGeometry();
geom.setAttribute('position',new THREE.Float32BufferAttribute(P,3));
geom.setAttribute('color',new THREE.Float32BufferAttribute(C,3));
scene.add(new THREE.Points(geom,new THREE.PointsMaterial({{size:{args.point_size},vertexColors:true,sizeAttenuation:true}})));
const cameraPoses={camera_pose_json};
const box=new THREE.Box3().setFromBufferAttribute(geom.getAttribute('position'));
const center=new THREE.Vector3();box.getCenter(center);
const sz=Math.max(box.getSize(new THREE.Vector3()).length(), 0.1);
const axisLen=Math.max(sz*0.08, 0.02);
function addLine(start, dir, len, color) {{
  const end = start.clone().add(dir.clone().multiplyScalar(len));
  const g = new THREE.BufferGeometry().setFromPoints([start, end]);
  const l = new THREE.Line(g, new THREE.LineBasicMaterial({{color}}));
  scene.add(l);
}}
function addCameraPose(pose) {{
  const pos = new THREE.Vector3(pose.position[0], pose.position[1], pose.position[2]);
  const m = new THREE.Mesh(new THREE.SphereGeometry(axisLen*0.18), new THREE.MeshBasicMaterial({{color: pose.color}}));
  m.position.copy(pos);
  scene.add(m);
  const R = pose.rotation_c2w;
  const xAxis = new THREE.Vector3(R[0][0], R[1][0], R[2][0]).normalize();
  const yAxis = new THREE.Vector3(R[0][1], R[1][1], R[2][1]).normalize();
  const zAxis = new THREE.Vector3(R[0][2], R[1][2], R[2][2]).normalize();
  addLine(pos, xAxis, axisLen, 0xff3333);
  addLine(pos, yAxis, axisLen, 0x33ff66);
  addLine(pos, zAxis, axisLen, 0x3399ff);
}}
cameraPoses.forEach(addCameraPose);
const ctl=new OrbitControls(camera,renderer.domElement);ctl.target.copy(center);
camera.position.set(center.x+sz*.5,center.y+sz*.3,center.z+sz*.5);
ctl.enableDamping=true;ctl.update();
const grid=new THREE.GridHelper(sz*1.5,20,0x336633,0x222222);
scene.add(grid);
scene.add(new THREE.AxesHelper(sz*0.15));
addEventListener('resize',()=>{{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)}});
(function r(){{requestAnimationFrame(r);ctl.update();renderer.render(scene,camera)}})();
</script></body></html>"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html)
    print(f"    -> {output_html}")

    stats = {
        "input_dir": str(input_dir),
        "image_count": len(image_paths),
        "process_res": args.process_res,
        "process_res_method": args.process_res_method,
        "depth_shape": list(depths.shape),
        "is_metric": bool(is_metric),
        "infer_time_sec": float(t_infer),
        "preview_max_points_per_image": int(args.max_points_per_image),
        "preview_total_points": int(len(preview_points_all)),
        "source_max_points_per_image": int(args.source_max_points_per_image),
        "source_total_points": int(len(source_points)) if source_points is not None else 0,
        "output_html": str(output_html),
        "output_source_npz": str(output_source_npz) if output_source_npz else None,
        "output_source_ply": str(output_source_ply) if output_source_ply else None,
        "per_image": per_image_stats,
    }

    if output_source_npz and source_points is not None:
        output_source_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_source_npz,
            points=source_points.astype(np.float32),
            colors=source_colors.astype(np.float32),
            camera_positions=np.stack(cam_positions, axis=0).astype(np.float32),
            camera_rotations_c2w=np.stack(cam_rotations_c2w, axis=0).astype(np.float32),
            extrinsics_w2c=extrs.astype(np.float32),
            intrinsics=intrs.astype(np.float32),
            image_names=np.array([Path(p).name for p in image_paths]),
            process_res=np.int32(args.process_res),
            is_metric=np.int32(1 if is_metric else 0),
        )
        print(f"[8] 写出 source NPZ -> {output_source_npz}")

    if output_source_ply and source_points is not None:
        write_ply(output_source_ply, source_points, source_colors)
        print(f"[9] 写出 source PLY -> {output_source_ply}")

    if output_stats:
        output_stats.parent.mkdir(parents=True, exist_ok=True)
        output_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f"[10] 写出 stats -> {output_stats}")

    print("\n完成！")


if __name__ == "__main__":
    main()
