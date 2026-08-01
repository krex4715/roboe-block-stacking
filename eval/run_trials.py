"""[ROBOE] M6 - 인식 기반 스태킹 배치 평가 (N회 랜덤 스폰, 성공률/시간/브리지 통계).

**한 프로세스에서 LOAD 를 반복**한다 (트라이얼마다 SimulationApp 재부팅 대비 ~10초씩 절약).
이게 가능한 것 자체가 M2에서 고친 재로드 버그(월드 싱글톤 재사용) 수정의 실전 검증이다.

**트라이얼 설계의 핵심**: 물리 큐브만 랜덤 위치로 옮기고 **고스트(belief)는 기본
스폰 위치에 그대로 둔다.** 즉 매 트라이얼이 "시작부터 belief 가 틀린 상태"에서 출발하고,
인식이 그 오차를 스스로 교정해야만 성공한다. 인식이 죽어 있으면 로봇은 큐브가 없는
기본 위치로 손을 뻗는다 - 성공률이 곧 인식 기여의 증명이다.

스폰 제약 (behavior 코드에서 역산):
  - 로봇 베이스에서 0.30~0.75 m (픽 유효 반경 0.25~0.81 에 여유)
  - 적재 위치에서 0.16 m 초과 (탑 보호/픽 거부 반경 밖)
  - 큐브 상호 간격 > 1.6변 (겹침 방지)
  - 카메라 시야 안 (SDG 와 같은 산란 영역)

실행:
    python eval/run_trials.py --trials 10
    python eval/run_trials.py --trials 3 --record   # 첫 트라이얼 카메라 영상 저장
"""

import argparse
import csv
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=10)
parser.add_argument("--tower", type=float, nargs=2, default=[0.25, 0.30], metavar=("X", "Y"))
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max-steps", type=int, default=6000, help="트라이얼당 타임아웃 (물리 스텝, 완주는 ~1400)")
parser.add_argument("--record", action="store_true", help="첫 트라이얼의 카메라 프레임을 mp4 로 저장")
parser.add_argument("--out", type=str, default=str(REPO / "media" / "m6"))
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import asyncio  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.examples.browser")
enable_extension("isaacsim.examples.interactive")
for _ in range(5):
    simulation_app.update()

from isaacsim.examples.interactive.user_examples.roboe_block_stacking.roboe_stacking_example import (  # noqa: E402
    RoboeBlockStacking,
)
from isaacsim.examples.interactive.user_examples.roboe_block_stacking.scene_setup import (  # noqa: E402
    CUBE_HALF,
    CUBE_SIZE,
)

BEHAVIOR_PATH = str(REPO / "isaac_ext" / "roboe_block_stacking" / "behavior" / "block_stacking_behavior.py")
EXPECTED_ORDER = ["RedCube", "YellowCube", "GreenCube", "BlueCube"]
XY_TOL = 0.04
SCATTER_X, SCATTER_Y = (0.28, 0.72), (-0.55, 0.38)
MIN_GAP = CUBE_SIZE * 1.6

outdir = Path(args.out)
outdir.mkdir(parents=True, exist_ok=True)


def pump(coro, label, limit=6000):
    fut = asyncio.ensure_future(coro)
    for _ in range(limit):
        if fut.done():
            break
        simulation_app.update()
    if not fut.done():
        raise TimeoutError(label)
    fut.result()


def sample_spawns(rng, tower_xy):
    """제약을 만족하는 랜덤 스폰 4개 (기각 샘플링)."""
    pts = []
    for _ in range(2000):
        if len(pts) == 4:
            break
        p = np.array([rng.uniform(*SCATTER_X), rng.uniform(*SCATTER_Y)])
        r = np.linalg.norm(p)
        if not (0.30 <= r <= 0.75):
            continue
        if np.linalg.norm(p - tower_xy) <= 0.16:
            continue
        if any(np.linalg.norm(p - q) <= MIN_GAP for q in pts):
            continue
        pts.append(p)
    while len(pts) < 4:  # 극히 드묾
        pts.append(np.array([0.30 + 0.13 * len(pts), -0.45]))
    return pts


def evaluate(cubes, tower_xy):
    rows = [(n, np.asarray(o.get_world_pose()[0], dtype=float)) for n, o in cubes.items()]
    stacked = sorted(rows, key=lambda r: r[1][2])
    order = [n for n, _ in stacked]
    max_off = max(float(np.linalg.norm(p[:2] - tower_xy)) for _, p in rows)
    return order == EXPECTED_ORDER and max_off <= XY_TOL, order, max_off


def main():
    tower_xy = np.asarray(args.tower, dtype=float)
    rng = np.random.default_rng(args.seed)

    sample = RoboeBlockStacking(perception_fn=lambda t: None)
    sample.behavior = BEHAVIOR_PATH
    sample.belief_mode = "ghost"
    sample.tower_position = np.array([tower_xy[0], tower_xy[1], 0.0])

    results = []
    for trial in range(args.trials):
        spawns = sample_spawns(rng, tower_xy)
        t0 = time.time()
        pump(sample.load_world_async(), f"trial{trial}:LOAD")

        # 물리 큐브만 랜덤 배치. 고스트는 기본 위치 그대로 = belief 를 의도적으로 틀리게 시작.
        for (name, obj), p in zip(sample.cubes.items(), spawns):
            yaw = rng.uniform(0, np.pi / 2)
            obj.set_world_pose(position=np.array([p[0], p[1], CUBE_HALF]),
                               orientation=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))

        init_belief_err = [
            float(np.linalg.norm(np.asarray(sample.belief_cubes[n].get_world_pose()[0])[:2]
                                 - np.asarray(sample.cubes[n].get_world_pose()[0])[:2]))
            for n in sample.cubes
        ]

        writer = None
        if args.record and trial == 0:
            writer = {"frames": 0, "vw": None}

        pump(sample.on_event_async(), f"trial{trial}:Start")
        world = sample.get_world()
        ctx = sample.decider_network.context

        done_at = None
        for step in range(args.max_steps):
            world.step(render=True)
            if writer is not None and step % 6 == 0 and sample.zed is not None:
                rgb, _ = sample.zed.capture()
                if rgb is not None:
                    if writer["vw"] is None:
                        h, w = rgb.shape[:2]
                        writer["vw"] = cv2.VideoWriter(
                            str(outdir / "trial0_zed.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
                    writer["vw"].write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    writer["frames"] += 1
            if ctx.block_tower.is_complete:
                if done_at is None:
                    done_at = step
                if step > done_at + 150:
                    break
        if writer is not None and writer["vw"] is not None:
            writer["vw"].release()
            print(f"    [record] {writer['frames']} 프레임 -> {outdir/'trial0_zed.mp4'}")

        ok, order, max_off = evaluate(sample.cubes, tower_xy)
        elapsed = time.time() - t0
        stats = dict(sample.bridge.stats)
        results.append({
            "trial": trial, "success": ok, "steps": step, "wall_s": round(elapsed, 1),
            "behavior_done": done_at is not None, "max_xy_off_mm": round(max_off * 1000, 1),
            "init_belief_err_mm": round(float(np.mean(init_belief_err)) * 1000, 1),
            "order": "->".join(order),
            "grasps": stats.get("grasp_events"), "ghost_rejects": stats.get("gated_workspace"),
            "published": stats.get("published"), "snapped": stats.get("snapped"),
            "reacquired": stats.get("reacquired"),
        })
        # 트라이얼마다 즉시 저장 - 전체 타임아웃/정전에도 결과 보존
        csv_path = outdir / "trials.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        r = results[-1]
        print(f"[trial {trial}] {'PASS' if ok else 'FAIL':4s} steps={r['steps']} "
              f"wall={r['wall_s']}s 초기belief오차={r['init_belief_err_mm']}mm "
              f"파지={r['grasps']} 유령기각={r['ghost_rejects']}", flush=True)

    csv_path = outdir / "trials.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    n_ok = sum(r["success"] for r in results)
    walls = [r["wall_s"] for r in results if r["success"]]
    print(f"\n성공률: {n_ok}/{len(results)} ({n_ok/len(results)*100:.0f}%)")
    if walls:
        print(f"완주 시간: 평균 {np.mean(walls):.1f}s / 최소 {min(walls):.1f}s / 최대 {max(walls):.1f}s")
    print(f"초기 belief 오차(평균): {np.mean([r['init_belief_err_mm'] for r in results]):.0f}mm "
          f"-> 인식이 매 트라이얼 이만큼을 스스로 교정해야 성공")
    print(f"결과 CSV: {csv_path}")
    print(f"M6_RESULT: {'PASS' if n_ok >= len(results) * 0.9 else 'FAIL'} (기준: 성공률 >= 90%)")


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
    print("M6_RESULT: FAIL (예외)")
    sys.stdout.flush()
    sys.stderr.flush()
finally:
    simulation_app.close()
