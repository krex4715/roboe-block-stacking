"""[ROBOE] M4 검증 - Isaac Sim 안에서 YOLO 추론 + 3D 위치 추정 정확도.

M1은 **GT 픽셀**을 써서 기하 체인만 검증했다. 여기서는 그 픽셀을 **AI 검출 결과**로 바꾼다.
즉 M1 대비 새로 들어오는 오차원은 두 가지다:
  1. 검출 bbox 중심 != 큐브 3D 중심의 투영 (실루엣 중심과 기하 중심의 차이)
  2. 검출 자체의 실패/혼동

두 오차를 분리해 보려고 M1과 **같은 보정 3종(none/ray/box)** 으로 함께 측정한다.

실행:
    python standalone/verify_m4_perception.py --frames 30
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / "isaac_ext" / "roboe_block_stacking"

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=30, help="측정할 랜덤 배치 수")
parser.add_argument("--model", type=str, default=str(REPO / "models" / "best.torchscript"))
parser.add_argument("--conf", type=float, default=0.5)
parser.add_argument("--outdir", type=str, default=str(REPO / "media" / "m4"))
parser.add_argument("--settle", type=int, default=12, help="배치 변경 후 렌더 안정화 프레임")
parser.add_argument("--gui", action="store_true")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.gui})

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402

sys.path.insert(0, str(EXT))
from perception.detector import CubeDetector  # noqa: E402
from perception.estimator_3d import (  # noqa: E402
    backproject_pixels,
    clamp_to_support,
    sample_depth,
    surface_to_center,
)
from perception.zed_camera import ZedXCamera  # noqa: E402
from scene_setup import (  # noqa: E402
    CUBE_HALF,
    CUBE_SIZE,
    CUBE_SPECS,
    add_cubes,
    add_lighting,
    add_tower_marker,
)

CLASS_TO_PRIM = {"red_cube": "RedCube", "yellow_cube": "YellowCube",
                 "green_cube": "GreenCube", "blue_cube": "BlueCube"}
SCATTER_X, SCATTER_Y = (0.28, 0.72), (-0.55, 0.38)
MIN_GAP = CUBE_SIZE * 1.6

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)


def scatter(rng, n):
    pts = []
    for _ in range(400):
        if len(pts) == n:
            break
        p = np.array([rng.uniform(*SCATTER_X), rng.uniform(*SCATTER_Y)])
        if all(np.linalg.norm(p - q) > MIN_GAP for q in pts):
            pts.append(p)
    while len(pts) < n:
        pts.append(np.array([SCATTER_X[0] + 0.12 * len(pts), SCATTER_Y[0]]))
    return pts


def draw_overlay(rgb, dets, estimates, path):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    palette = {"red_cube": (60, 60, 220), "yellow_cube": (60, 220, 220),
               "green_cube": (60, 220, 120), "blue_cube": (220, 100, 60)}
    for d in dets:
        x0, y0, x1, y1 = [int(v) for v in d["box"]]
        c = palette.get(d["class"], (200, 200, 200))
        cv2.rectangle(img, (x0, y0), (x1, y1), c, 2)
        label = f"{d['class']} {d['score']:.2f}"
        cv2.putText(img, label, (x0, y0 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        est = estimates.get(d["class"])
        if est is not None:
            cv2.putText(img, f"({est[0]:.2f},{est[1]:.2f},{est[2]:.2f})",
                        (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    cv2.imwrite(str(path), img)


def main():
    if not Path(args.model).exists():
        print(f"M4_GATE: FAIL - 모델 없음: {args.model}")
        return
    meta_path = REPO / "models" / "model_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    class_names = meta.get("class_names", list(CLASS_TO_PRIM))
    imgsz = int(meta.get("imgsz", 640))

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_lighting()
    cubes = add_cubes(world.scene)
    add_tower_marker(np.array([0.25, 0.30, 0.0]), world.scene)
    try:
        from isaacsim.cortex.framework.robot import add_franka_to_stage

        world.scene.add(add_franka_to_stage(name="franka", prim_path="/World/Franka"))
    except Exception as exc:
        print(f"[scene] Franka 추가 실패: {exc}")

    zed = ZedXCamera()
    zed.spawn()
    world.reset()
    zed.initialize()
    detector = CubeDetector(args.model, class_names, imgsz=imgsz, conf=args.conf)
    print(zed.describe(), flush=True)

    rng = np.random.default_rng(1234)
    rows = []
    stats = {"frames": 0, "expected": 0, "detected": 0, "wrong_class": 0, "missed": 0}
    latency = []

    for frame in range(args.frames):
        # 큐브를 랜덤 배치 (SDG 와 같은 분포, 단 시드가 다르다)
        pts = scatter(rng, len(CUBE_SPECS))
        for (name, _), p in zip(CUBE_SPECS, pts):
            yaw = rng.uniform(0, np.pi / 2)
            cubes[name].set_world_pose(
                position=np.array([p[0], p[1], CUBE_HALF]),
                orientation=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
            )
            cubes[name].set_linear_velocity(np.zeros(3))
            cubes[name].set_angular_velocity(np.zeros(3))

        for _ in range(args.settle):
            world.step(render=True)
        rgb, depth = zed.capture()
        if rgb is None or depth is None:
            continue

        dets = detector(rgb)
        latency.append(detector.last_latency_ms)
        best = detector.best_per_class(dets)
        stats["frames"] += 1
        stats["expected"] += len(cubes)
        stats["detected"] += len(best)

        cam_pos = zed.position
        estimates = {}
        for cls, prim_name in CLASS_TO_PRIM.items():
            gt = np.asarray(cubes[prim_name].get_world_pose()[0], dtype=float)
            d = best.get(cls)
            if d is None:
                stats["missed"] += 1
                rows.append({"frame": frame, "class": cls, "status": "missed"})
                continue
            x0, y0, x1, y1 = d["box"]
            u, v = (x0 + x1) / 2, (y0 + y1) / 2
            depth_val = sample_depth(depth, (u, v), box=(x0, y0, x1, y1))
            if depth_val is None:
                rows.append({"frame": frame, "class": cls, "status": "no_depth"})
                continue

            p_surf = backproject_pixels(zed.camera, [[u, v]], [depth_val])[0]
            row = {"frame": frame, "class": cls, "status": "ok", "score": d["score"],
                   "u": u, "v": v, "depth": depth_val,
                   "gt_x": gt[0], "gt_y": gt[1], "gt_z": gt[2]}
            for mode in ("none", "ray", "box"):
                p = clamp_to_support([surface_to_center([p_surf], cam_pos, mode=mode)[0]])[0]
                row[f"err_{mode}"] = float(np.linalg.norm(p - gt))
                if mode == "box":
                    row.update({"est_x": p[0], "est_y": p[1], "est_z": p[2]})
                    row["err_xy"] = float(np.linalg.norm(p[:2] - gt[:2]))
                    row["err_z"] = float(abs(p[2] - gt[2]))
                    estimates[cls] = p
            rows.append(row)

        if frame < 6:
            draw_overlay(rgb, dets, estimates, outdir / f"overlay_{frame:02d}.jpg")

    # ------------------------------------------------------------------ 결과
    csv_path = outdir / "perception_error.csv"
    keys = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"\n프레임 {stats['frames']} / 기대 {stats['expected']} / 검출 {stats['detected']} "
          f"/ 미검출 {stats['missed']}")
    if latency:
        print(f"추론 지연: 중앙 {np.median(latency):.2f}ms / 최대 {np.max(latency):.2f}ms")

    if ok:
        print(f"\n{'보정':8s} {'평균':>9s} {'중앙':>9s} {'p95':>9s} {'최대':>9s}   (mm)")
        print("-" * 52)
        for mode in ("none", "ray", "box"):
            e = np.array([r[f"err_{mode}"] for r in ok]) * 1000
            print(f"{mode:8s} {e.mean():8.1f} {np.median(e):8.1f} "
                  f"{np.percentile(e,95):8.1f} {e.max():8.1f}")
        eb = np.array([r["err_box"] for r in ok]) * 1000
        exy = np.array([r["err_xy"] for r in ok]) * 1000
        ez = np.array([r["err_z"] for r in ok]) * 1000
        print(f"\nbox 보정 축별: xy 평균 {exy.mean():.1f}mm / z 평균 {ez.mean():.1f}mm")
        print(f"클래스별 평균 오차(box):")
        for cls in CLASS_TO_PRIM:
            e = [r["err_box"] * 1000 for r in ok if r["class"] == cls]
            if e:
                print(f"    {cls:12s} {np.mean(e):6.1f}mm  (n={len(e)})")

        passed = (stats["missed"] == 0 and eb.mean() < 10.0 and np.percentile(eb, 95) < 20.0)
        print(f"\n결과 CSV: {csv_path}")
        print(f"M4_GATE: {'PASS' if passed else 'FAIL'} "
              f"(기준: 미검출 0 + 평균오차<10mm + p95<20mm)")
    else:
        print("M4_GATE: FAIL - 유효 측정 없음")


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
finally:
    simulation_app.close()
