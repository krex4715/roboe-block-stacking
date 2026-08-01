"""[ROBOE] 제출용 데모 영상 레코더 - [3인칭 뷰 | ZED + YOLO 오버레이] 나란히.

시나리오 (배치 평가와 같은 스트레스 조건):
  1. 큐브를 파지 유효 작업공간 안에 랜덤 위치+yaw 로 스폰
  2. 인식 기반 스태킹 시작 (ghost belief - 로봇은 인식 결과만 본다)
  3. 탑이 2층이 되면 남은 큐브를 **라이브 랜덤 재배치** (GUI 의 Randomize 버튼과
     같은 코드) -> 인식이 재검출로 따라잡아 계속 진행하는 반응성 데모
  4. 완성 후 2초 유지

출력: media/demo/demo_stacking.mp4 (2560x720, 30fps = 실시간 x1)
실행: python standalone/record_demo.py [--seed 7]
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--max-steps", type=int, default=6000)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import asyncio  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402

enable_extension("isaacsim.examples.browser")
enable_extension("isaacsim.examples.interactive")
for _ in range(5):
    simulation_app.update()

from isaacsim.sensors.camera import Camera  # noqa: E402

from isaacsim.examples.interactive.user_examples.roboe_block_stacking.roboe_stacking_example import (  # noqa: E402
    RoboeBlockStacking,
)

BEHAVIOR = str(REPO / "isaac_ext/roboe_block_stacking/behavior/block_stacking_behavior.py")
CUBE_SIZE = 0.0515
MIN_GAP = CUBE_SIZE * 1.6
TOWER = np.array([0.25, 0.30])
W, H = 1280, 720
# 검출 뷰와 동일한 클래스 색 (RGB). 캡처 후 주석이므로 되먹임 불가.
COLORS = {"red_cube": (255, 60, 60), "yellow_cube": (255, 220, 40),
          "green_cube": (140, 230, 60), "blue_cube": (70, 130, 255)}


def pump(coro, limit=6000):
    fut = asyncio.ensure_future(coro)
    for _ in range(limit):
        if fut.done():
            break
        simulation_app.update()
    fut.result()


def sample_spawns(rng):
    """배치 평가와 같은 명세: 파지 유효 작업공간 r [0.40, 0.75], 탑/상호 간섭 회피."""
    pts = []
    for _ in range(2000):
        if len(pts) == 4:
            break
        p = np.array([rng.uniform(0.28, 0.72), rng.uniform(-0.55, 0.38)])
        if not (0.40 <= np.linalg.norm(p) <= 0.75):
            continue
        if np.linalg.norm(p - TOWER) <= 0.16:
            continue
        if any(np.linalg.norm(p - q) <= MIN_GAP for q in pts):
            continue
        pts.append(p)
    return pts


def look_at_quat(eye, target):
    """x-전방/z-상방(world axes) 카메라용 look-at 자세."""
    f = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    f /= np.linalg.norm(f)
    yaw = np.arctan2(f[1], f[0])
    pitch = -np.arcsin(f[2])
    return euler_angles_to_quat(np.array([0.0, pitch, yaw]))


def annotate_zed(rgb, detections):
    img = np.ascontiguousarray(rgb[..., :3]).copy()
    for d in detections:
        x0, y0, x1, y1 = (int(v) for v in d["box"])
        c = COLORS.get(d["class"], (255, 255, 255))
        cv2.rectangle(img, (x0, y0), (x1, y1), c, 3)
        cv2.putText(img, f"{d['class']} {d['score']:.2f}", (x0, max(y0 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    cv2.putText(img, "ZED-X view + YOLOv8n (10 Hz, in-the-loop)", (14, H - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def annotate_spectator(rgba, caption):
    img = np.ascontiguousarray(rgba[..., :3]).copy()
    cv2.putText(img, "ROBOE Block Stacking - perception-driven (ghost belief)",
                (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if caption:
        cv2.putText(img, caption, (14, H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    rng = np.random.default_rng(args.seed)
    sample = RoboeBlockStacking(perception_fn=lambda t: None)
    sample.behavior = BEHAVIOR
    sample.belief_mode = "ghost"
    sample.tower_position = np.array([TOWER[0], TOWER[1], 0.0])
    pump(sample.load_world_async())

    # 랜덤 스폰 (배치 평가 조건)
    for (name, obj), p in zip(sample.cubes.items(), sample_spawns(rng)):
        yaw = rng.uniform(0, np.pi / 2)
        obj.set_world_pose(position=np.array([p[0], p[1], CUBE_SIZE / 2]),
                           orientation=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        obj.set_linear_velocity(np.zeros(3))
        obj.set_angular_velocity(np.zeros(3))

    # 3인칭 카메라 - 스폰 영역(-y) 쪽에서 로봇+작업공간 전체를 담는다.
    # 주의: ZED-X 마운트가 (1.0, 0.0, 1.0) 에 있으므로 시선축에서 30도 이상
    # 비켜난 위치를 골라야 화면에 걸리지 않는다 (첫 촬영에서 실측한 제약).
    eye = np.array([0.35, -1.75, 1.25])
    spect = Camera(prim_path="/World/DemoCam", position=eye,
                   orientation=look_at_quat(eye, np.array([0.40, 0.05, 0.05])),
                   resolution=(W, H))

    pump(sample.on_event_async())
    spect.initialize()
    world = sample.get_world()
    ctx = sample.decider_network.context

    frames_dir = Path(tempfile.mkdtemp(prefix="roboe_demo_"))
    out_dir = REPO / "media" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_frame = 0
    randomized_at = None
    caption, caption_until = "", 0
    done_at = None
    for step in range(args.max_steps):
        world.step(render=True)

        # 탑 2층 + 그리퍼 빈 시점에 남은 큐브 라이브 재배치 (반응성 데모)
        if randomized_at is None and ctx.block_tower.height >= 2 \
                and getattr(ctx, "in_gripper", None) is None:
            report = sample.randomize_cubes()
            randomized_at = step
            caption, caption_until = "LIVE RANDOMIZE - belief lags, perception re-acquires", step + 240
            print(f"[demo] {report} @step {step}", flush=True)

        if step % 2 == 0:  # 60Hz 물리 -> 30fps = 실시간 배속
            rgb, _ = sample.zed.capture()
            spect_rgba = spect.get_rgba()
            if rgb is None or spect_rgba is None or spect_rgba.size == 0:
                continue
            left = annotate_spectator(spect_rgba, caption if step < caption_until else "")
            right = annotate_zed(rgb, sample.last_detections)
            frame = cv2.hconcat([left, right])
            cv2.imwrite(str(frames_dir / f"f_{n_frame:05d}.jpg"),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            n_frame += 1

        if ctx.block_tower.is_complete:
            if done_at is None:
                done_at = step
                caption, caption_until = "TOWER COMPLETE (Red-Yellow-Green-Blue)", step + 10**9
            if step > done_at + 120:  # 완성 후 2초 유지
                break

    print(f"[demo] 프레임 {n_frame}장, 인코딩 시작", flush=True)
    out = out_dir / "demo_stacking.mp4"
    subprocess.run(["ffmpeg", "-y", "-framerate", "30",
                    "-i", str(frames_dir / "f_%05d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
                   check=True, capture_output=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    order = [n for n, _ in sorted(((n, np.asarray(o.get_world_pose()[0])) for n, o in sample.cubes.items()),
                                  key=lambda r: r[1][2])]
    print(f"DEMO_RESULT: {'PASS' if done_at else 'FAIL'} 순서={order} -> {out}", flush=True)


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
finally:
    sys.stdout.flush()
    simulation_app.close()
