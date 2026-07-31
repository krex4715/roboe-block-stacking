"""[ROBOE] M5 게이트 - 인식 기반 end-to-end 스태킹 (GUI 예제와 **같은 코드 경로**).

`run_stacking.py` 는 씬을 직접 만들지만, 이 스크립트는 GUI 예제 클래스
`RoboeBlockStacking` 을 그대로 태운다 (LOAD -> Start -> 물리 스텝).
즉 여기서 통과하면 GUI 에서도 같은 동작이 나온다.

성공 판정은 **물리 큐브**(고스트 아님)의 최종 좌표로 한다 —
로봇이 믿는 것과 실제로 쌓인 것은 다른 문제이므로.

실행:
    python standalone/run_stacking_perception.py                 # ghost + 인식 ON
    python standalone/run_stacking_perception.py --belief direct # 비교군(ground truth)
    python standalone/run_stacking_perception.py --no-perception # 인식 OFF (고스트 정지)
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--belief", choices=["ghost", "direct"], default="ghost")
parser.add_argument("--no-perception", action="store_true")
parser.add_argument("--tower", type=float, nargs=2, default=[0.25, 0.30], metavar=("X", "Y"))
parser.add_argument("--max-steps", type=int, default=16000)
parser.add_argument("--gui", action="store_true")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.gui})

import asyncio  # noqa: E402

import numpy as np  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.examples.browser")
enable_extension("isaacsim.examples.interactive")
for _ in range(5):
    simulation_app.update()

from isaacsim.examples.interactive.user_examples.roboe_block_stacking.roboe_stacking_example import (  # noqa: E402
    RoboeBlockStacking,
)

BEHAVIOR_PATH = str(REPO / "isaac_ext" / "roboe_block_stacking" / "behavior" / "block_stacking_behavior.py")
EXPECTED_ORDER = ["RedCube", "YellowCube", "GreenCube", "BlueCube"]
XY_TOL = 0.04


def pump(coro, label, limit=6000):
    fut = asyncio.ensure_future(coro)
    for _ in range(limit):
        if fut.done():
            break
        simulation_app.update()
    if not fut.done():
        raise TimeoutError(label)
    fut.result()


def evaluate(cubes, tower_xy):
    rows = []
    for name, obj in cubes.items():
        p = np.asarray(obj.get_world_pose()[0], dtype=float)
        rows.append((name, p, float(np.linalg.norm(p[:2] - np.asarray(tower_xy)))))
    stacked = sorted(rows, key=lambda r: r[1][2])
    order = [n for n, _, _ in stacked]
    off = [(n, d) for n, _, d in rows if d > XY_TOL]

    print(f"\n{'큐브':11s} {'최종 위치 (m)':32s} {'탑 중심 거리':>12s}")
    print("-" * 60)
    for name, p, d in stacked:
        flag = "" if d <= XY_TOL else "  <- 탑에서 벗어남"
        print(f"{name:11s} [{p[0]:6.3f} {p[1]:6.3f} {p[2]:6.3f}]{'':>12s} {d:8.3f}m{flag}")
    print("-" * 60)
    print(f"실제 순서(아래->위): {' -> '.join(order)}")
    print(f"기대 순서(아래->위): {' -> '.join(EXPECTED_ORDER)}")
    return (order == EXPECTED_ORDER) and not off


def main():
    sample = RoboeBlockStacking(perception_fn=lambda t: None)
    sample.behavior = BEHAVIOR_PATH
    sample.belief_mode = args.belief
    sample.tower_position = np.array([args.tower[0], args.tower[1], 0.0])
    sample.perception_enabled = not args.no_perception

    print(f"[setup] belief={args.belief}  perception={'OFF' if args.no_perception else 'ON'}  "
          f"tower={sample.tower_position[:2]}")

    pump(sample.load_world_async(), "LOAD")
    print(f"[setup] 검출기 {'로드됨' if sample.detector else '없음'} / "
          f"물리큐브 {len(sample.cubes)} / 고스트 {len(sample.belief_cubes)}")
    print(f"[setup] behavior 가 보는 대상: {list(sample.robot.registered_obstacles.keys())}")

    pump(sample.on_event_async(), "Start")
    world = sample.get_world()
    ctx = sample.decider_network.context

    t0, done_at = time.time(), None
    for step in range(args.max_steps):
        world.step(render=True)
        if ctx.block_tower.is_complete:
            if done_at is None:
                done_at = step
                print(f"[run] behavior 완료 보고 ({step} 스텝)", flush=True)
            if step > done_at + 150:
                break
        if step % 1000 == 0 and step:
            # 막혔을 때 '어디서' 막혔는지 알려면 의사결정 스택 + 세계 상태가 필요하다
            stack = " > ".join(str(e) for e in sample.decider_network._decider_state.stack)
            print(f"[run] {step} 스텝... 탑 높이={ctx.block_tower.height} "
                  f"| {sample.bridge.summary()}\n      결정스택: {stack}", flush=True)
            for name in ["RedCube", "YellowCube", "GreenCube", "BlueCube"]:
                g = np.asarray(sample.belief_cubes[name].get_world_pose()[0]) if sample.belief_cubes else None
                r = np.asarray(sample.cubes[name].get_world_pose()[0])
                reason = sample.bridge.last_reasons.get(name, "-")
                cls = {"RedCube": "red_cube", "YellowCube": "yellow_cube",
                       "GreenCube": "green_cube", "BlueCube": "blue_cube"}[name]
                est = sample.last_estimates.get(cls)
                print(f"        {name:11s} ghost={np.round(g,3) if g is not None else '-'} "
                      f"real={np.round(r,3)} est={np.round(est,3) if est is not None else '없음'} "
                      f"| {reason}", flush=True)

    elapsed = time.time() - t0
    print(f"\n[bridge] {sample.bridge.summary()}")
    ok = evaluate(sample.cubes, args.tower)
    print(f"\n스텝 {step} (완료보고 {done_at}) / {elapsed:.1f}s")
    print(f"M5_GATE: {'PASS' if (ok and done_at is not None) else 'FAIL'}")


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
    print("M5_GATE: FAIL (예외)")
    sys.stdout.flush()
    sys.stderr.flush()
finally:
    simulation_app.close()
