"""[ROBOE] M1 검증 - ZED-X 스폰 + RGB-D 취득 + 3D 역투영 기하 체인 검증.

AI를 붙이기 **전에** 기하 체인이 맞는지부터 확정한다. 검출기가 틀린 건지 좌표변환이 틀린 건지
나중에 구분할 수 없게 되는 것을 막기 위한 게이트다.

왕복(round-trip) 테스트:
    큐브 GT 중심 --투영--> 픽셀 --깊이 읽기--> 역투영 --> 표면점 --보정--> 중심 추정
    추정 중심 vs GT 중심 오차가 ~1cm 이내면 통과.

보정 3종(none/ray/box)을 동시에 비교해 "왜 보정이 필요한가"를 수치로 남긴다.

실행:
    conda activate isaacsim_roboe
    python standalone/verify_m1_camera.py            # 헤드리스
    python standalone/verify_m1_camera.py --gui      # 창 띄우기
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true", help="GUI 창을 띄운다")
parser.add_argument("--warmup", type=int, default=60, help="첫 유효 프레임까지 최대 대기 프레임")
parser.add_argument("--settle", type=int, default=40, help="디노이저 수렴용 추가 렌더 프레임")
parser.add_argument("--outdir", type=str, default=str(REPO / "media" / "m1"))
args = parser.parse_args()

# SimulationApp 은 다른 isaacsim import 보다 먼저 생성되어야 한다
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.gui})

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402


# perception 패키지를 top-level 로 import (GUI 확장 경로를 거치지 않기 위해)
sys.path.insert(0, str(REPO / "isaac_ext" / "roboe_block_stacking"))
from perception.estimator_3d import (  # noqa: E402
    CUBE_HALF,
    CUBE_SIZE,
    backproject_pixels,
    clamp_to_support,
    sample_depth,
    surface_to_center,
)
from perception.zed_camera import ZedXCamera  # noqa: E402
from scene_setup import add_cubes, add_lighting  # noqa: E402

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)


def build_scene(world):
    """베이스 예제와 같은 씬 (지면 + 큐브 4개 + Franka) + 명시적 조명."""
    world.scene.add_default_ground_plane()
    add_lighting()
    cubes = add_cubes(world.scene)
    try:
        from isaacsim.cortex.framework.robot import add_franka_to_stage

        world.scene.add(add_franka_to_stage(name="franka", prim_path="/World/Franka"))
        print("[scene] Franka 추가됨")
    except Exception as exc:
        print(f"[scene] Franka 추가 실패(검증에는 영향 없음): {exc}")
    return cubes


def save_images(rgb, depth, marks):
    """RGB / 깊이 컬러맵 / 검증 마커 오버레이 저장."""
    cv2.imwrite(str(outdir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    finite = np.isfinite(depth) & (depth > 0) & (depth < 1e4)
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if finite.any():
        lo, hi = np.percentile(depth[finite], [1, 99])
        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        vis = (norm * 255).astype(np.uint8)
        vis[~finite] = 0
    cv2.imwrite(str(outdir / "depth.png"), cv2.applyColorMap(vis, cv2.COLORMAP_TURBO))

    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for name, (u, v), ok in marks:
        color = (0, 255, 0) if ok else (0, 0, 255)
        ui, vi = int(round(u)), int(round(v))
        cv2.drawMarker(overlay, (ui, vi), color, cv2.MARKER_CROSS, 30, 2)
        cv2.putText(overlay, name, (ui + 14, vi - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(str(outdir / "overlay.png"), overlay)


def report_color_separability(rgb, marks):
    """렌더된 픽셀에서 큐브 색이 서로 충분히 구분되는지 측정.

    조명이 세면 채널이 255 근처로 포화되며 색상(hue) 정보가 뭉개진다. 특히 노랑/연두처럼
    색상환에서 가까운 쌍은 바로 오분류로 이어지므로, 눈으로 보지 말고 수치로 감시한다.
    """
    means, hsvs = {}, {}
    for name, (u, v), ok in marks:
        if not ok:
            continue
        ui, vi = int(round(u)), int(round(v))
        patch = rgb[max(0, vi - 6) : vi + 7, max(0, ui - 6) : ui + 7].reshape(-1, 3).astype(float)
        if patch.size == 0:
            continue
        means[name] = patch.mean(axis=0)
        hsvs[name] = cv2.cvtColor(np.uint8([[means[name]]]), cv2.COLOR_RGB2HSV)[0, 0]

    if len(means) < 2:
        return
    print("\n[색 분리도] 큐브 중심 13x13 픽셀 평균")
    for name in means:
        h, s, v = hsvs[name]
        clip = (means[name] > 250).sum()
        warn = "  ⚠ 채널 포화" if clip else ""
        print(f"    {name:11s} RGB={np.round(means[name]).astype(int)}  hue={h*2:5.1f}° "
              f"sat={s:3d} val={v:3d}{warn}")

    names = list(means)
    pairs = [(a, b, float(np.linalg.norm(means[a] - means[b])))
             for i, a in enumerate(names) for b in names[i + 1:]]
    pairs.sort(key=lambda x: x[2])
    print(f"    가장 가까운 색 쌍: {pairs[0][0]} vs {pairs[0][1]} = {pairs[0][2]:.1f} / 255 "
          f"(가장 먼 쌍 {pairs[-1][2]:.1f})")
    # 임계 80: 노랑/연두 쌍은 명세상 색상환에서 가까울 수밖에 없어 실측 95가 우리 기준선이다.
    # 이보다 더 내려가면 조명이 색을 뭉갠 것이므로 회귀로 본다.
    if pairs[0][2] < 80:
        print("    ⚠ 경고: 최소 색 거리가 80 미만 - 오분류 위험. 조명/색상 조정 검토")


def main():
    world = World(stage_units_in_meters=1.0)
    cubes = build_scene(world)

    zed = ZedXCamera()
    zed.spawn()

    world.reset()
    zed.initialize()
    print(zed.describe(), flush=True)

    # annotator 는 몇 프레임 렌더한 뒤에야 데이터를 낸다
    rgb = depth = None
    for i in range(args.warmup):
        world.step(render=True)
        rgb, depth = zed.capture()
        if rgb is not None and depth is not None and np.isfinite(depth).any():
            print(f"[capture] {i+1} 프레임 만에 RGB-D 취득 성공", flush=True)
            break
    if rgb is None or depth is None:
        print("M1_GATE: FAIL - RGB-D 취득 실패")
        return

    # RTX 디노이저/광량 누적이 수렴할 시간을 준다. 첫 유효 프레임을 바로 쓰면 노이즈가 심해
    # 학습 데이터/검출 입력으로 부적합하다.
    for _ in range(args.settle):
        world.step(render=True)
    rgb, depth = zed.capture()
    print(f"[capture] 안정화 {args.settle} 프레임 추가 렌더 후 최종 취득", flush=True)

    h, w = depth.shape[:2]
    finite = np.isfinite(depth) & (depth > 0)
    print(f"[capture] rgb={rgb.shape} depth={depth.shape} 유한 깊이 비율={finite.mean():.1%}")
    if finite.any():
        print(f"[capture] 깊이 범위=[{depth[finite].min():.3f}, {depth[finite].max():.3f}]m")
    else:
        print("[capture] ⚠ 유한한 깊이가 하나도 없음 - 카메라가 빈 공간을 보고 있을 가능성")

    # 카메라가 실제로 어디를 보는지 (규약 추론 대신 시선 벡터를 직접 뽑는다)
    view = np.asarray(zed.camera.get_view_matrix_ros())
    R_wc = view[:3, :3].T          # world <- camera 회전
    print(f"[view] 카메라 광축(월드) = {np.round(R_wc[:, 2], 4)}  (ROS 규약: +Z가 광축)")
    print(f"[view] 이미지 우측(월드) = {np.round(R_wc[:, 0], 4)},  하단(월드) = {np.round(R_wc[:, 1], 4)}")

    save_images(rgb, depth, [])  # 이후 단계가 실패해도 이미지는 남도록 먼저 저장
    cam_pos = zed.position
    rows, marks, in_view = [], [], 0

    for name in cubes:
        gt = np.asarray(cubes[name].get_world_pose()[0], dtype=np.float64)
        uv = np.asarray(zed.camera.get_image_coords_from_world_points(gt.reshape(1, 3)))[0]
        u, v = float(uv[0]), float(uv[1])
        visible = (0 <= u < w) and (0 <= v < h)
        marks.append((name, (u, v), visible))
        if not visible:
            rows.append((name, gt, (u, v), None, {}))
            continue
        in_view += 1

        d = float(depth[int(round(v)), int(round(u))])  # 게이트는 단일 픽셀(가장 순수한 기하 테스트)
        if not np.isfinite(d) or d <= 0:
            d_med = sample_depth(depth, (u, v))
            if d_med is None:
                rows.append((name, gt, (u, v), None, {}))
                continue
            d = d_med

        p_surf = backproject_pixels(zed.camera, [[u, v]], [d])[0]
        errs = {}
        for mode in ("none", "ray", "box"):
            p = surface_to_center([p_surf], cam_pos, mode=mode)[0]
            p = clamp_to_support([p])[0]
            errs[mode] = (p, float(np.linalg.norm(p - gt)))
        rows.append((name, gt, (u, v), d, errs))

    save_images(rgb, depth, marks)
    report_color_separability(rgb, marks)

    # ---------------------------------------------------------------- 결과 출력
    print("\n" + "=" * 96)
    print(f"{'cube':11s} {'GT center (m)':26s} {'pixel(u,v)':17s} {'depth':7s} "
          f"{'none':>8s} {'ray':>8s} {'box':>8s}   (오차 mm)")
    print("-" * 96)
    ok_box = []
    for name, gt, (u, v), d, errs in rows:
        gt_s = f"[{gt[0]:6.3f} {gt[1]:6.3f} {gt[2]:6.3f}]"
        px_s = f"({u:7.1f},{v:6.1f})"
        if d is None:
            print(f"{name:11s} {gt_s:26s} {px_s:17s} {'-':>7s} {'화면 밖 또는 깊이 없음':>30s}")
            continue
        print(f"{name:11s} {gt_s:26s} {px_s:17s} {d:6.3f}m "
              f"{errs['none'][1]*1000:8.1f} {errs['ray'][1]*1000:8.1f} {errs['box'][1]*1000:8.1f}")
        ok_box.append(errs["box"][1])
    print("=" * 96)

    if ok_box:
        arr = np.array(ok_box)
        print(f"box 보정 오차: 평균 {arr.mean()*1000:.1f}mm / 최대 {arr.max()*1000:.1f}mm "
              f"({len(ok_box)}/{len(cubes)}개 큐브)")
    print(f"화면 안에 들어온 큐브: {in_view}/{len(cubes)}")
    print(f"이미지 저장: {outdir}")

    passed = in_view == len(cubes) and bool(ok_box) and np.max(ok_box) < 0.01
    print(f"\nM1_GATE: {'PASS' if passed else 'FAIL'} "
          f"(조건: 큐브 4개 모두 화면 안 + box 보정 오차 최대 < 10mm)")


try:
    main()
except Exception:
    # SimulationApp.close() 가 프로세스를 강제 종료해 예외가 삼켜지므로 먼저 출력한다
    import traceback

    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
finally:
    simulation_app.close()
