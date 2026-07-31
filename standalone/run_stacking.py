"""[ROBOE] GUI 없이 블록 스태킹을 실행하고 결과를 자동 판정하는 러너.

용도:
  1) M2 검증 - 순서(빨->노->연->파)와 사용자 지정 적재 위치가 실제로 지켜지는지
  2) M6 배치 평가 - 여러 번 반복해 성공률/소요시간 집계
  3) 데모 보험 - GUI 예제 등록이 깨져도 이 스크립트로 동작을 보일 수 있다

성공 판정은 behavior 내부 상태가 아니라 **큐브의 최종 월드 좌표**로 한다.
(behavior 가 "완료했다"고 믿는 것과 물리적으로 탑이 서 있는 것은 다른 문제이므로)

실행:
    python standalone/run_stacking.py                       # 기본 위치
    python standalone/run_stacking.py --tower 0.45 0.25     # 적재 위치 지정
    python standalone/run_stacking.py --gui                 # 창 띄우고 보기
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / "isaac_ext" / "roboe_block_stacking"

parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true", help="GUI 창을 띄운다")
parser.add_argument("--tower", type=float, nargs=2, default=[0.25, 0.30], metavar=("X", "Y"))
parser.add_argument("--max-steps", type=int, default=12000, help="타임아웃 (물리 스텝)")
parser.add_argument("--no-camera", action="store_true", help="ZED-X 스폰 생략 (더 빠름)")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.gui})

import numpy as np  # noqa: E402
from isaacsim.cortex.framework.cortex_utils import load_behavior_module  # noqa: E402
from isaacsim.cortex.framework.cortex_world import CortexWorld  # noqa: E402
from isaacsim.cortex.framework.robot import add_franka_to_stage  # noqa: E402

sys.path.insert(0, str(EXT))
from perception.zed_camera import ZedXCamera  # noqa: E402
from scene_setup import (  # noqa: E402
    CUBE_HALF,
    add_cubes,
    add_lighting,
    add_tower_marker,
    validate_tower_position,
)

BEHAVIOR_PATH = str(EXT / "behavior" / "block_stacking_behavior.py")
# 과제 명세 순서. 아래(z 작은 쪽)부터 이 순서로 쌓여 있어야 성공.
EXPECTED_ORDER = ["RedCube", "YellowCube", "GreenCube", "BlueCube"]
XY_TOLERANCE = 0.03  # 탑 중심에서 허용하는 수평 오차


def evaluate_stack(cubes, tower_position):
    """큐브 최종 좌표로 성공 여부를 판정. (ok, 리포트문자열) 반환."""
    tower_xy = np.asarray(tower_position, dtype=float)[:2]
    rows = []
    for name, obj in cubes.items():
        p = np.asarray(obj.get_world_pose()[0], dtype=float)
        rows.append((name, p, float(np.linalg.norm(p[:2] - tower_xy))))

    # z 오름차순 = 아래에서 위 순서
    stacked = sorted(rows, key=lambda r: r[1][2])
    actual_order = [name for name, _, _ in stacked]
    off_tower = [(n, d) for n, _, d in rows if d > XY_TOLERANCE]

    lines = ["", f"{'큐브':11s} {'최종 위치 (m)':30s} {'탑 중심과 거리':>12s}"]
    lines.append("-" * 58)
    for name, p, d in stacked:
        flag = "" if d <= XY_TOLERANCE else "  <-- 탑에서 벗어남"
        lines.append(f"{name:11s} [{p[0]:6.3f} {p[1]:6.3f} {p[2]:6.3f}]{'':>10s} {d:8.3f}m{flag}")
    lines.append("-" * 58)
    lines.append(f"실제 순서(아래->위): {' -> '.join(actual_order)}")
    lines.append(f"기대 순서(아래->위): {' -> '.join(EXPECTED_ORDER)}")

    ok = (actual_order == EXPECTED_ORDER) and not off_tower
    return ok, "\n".join(lines)


def main():
    tower_position = np.array([args.tower[0], args.tower[1], 0.0])
    ok, msg = validate_tower_position(tower_position)
    print(f"[setup] 적재 목표 위치 {tower_position[:2]} -> {msg}")
    if not ok:
        print("STACK_RESULT: FAIL - 적재 위치가 유효하지 않습니다")
        return

    world = CortexWorld()
    robot = world.add_robot(add_franka_to_stage(name="franka", prim_path="/World/Franka"))
    cubes = add_cubes(world.scene)
    for obj in cubes.values():
        robot.register_obstacle(obj)
    world.scene.add_default_ground_plane()
    add_lighting()
    add_tower_marker(tower_position, world.scene)

    zed = None
    if not args.no_camera:
        zed = ZedXCamera()
        zed.spawn()

    decider_network = load_behavior_module(BEHAVIOR_PATH).make_decider_network(
        robot, tower_position=tower_position
    )
    world.add_decider_network(decider_network)
    print(f"[setup] 쌓기 순서: {decider_network.context.block_tower.desired_stack}")

    world.reset()
    if zed is not None:
        zed.initialize()
        print(f"[setup] ZED-X: {zed.info.get('camera_prim_path')}")

    state = {"steps": 0, "t0": time.time(), "done_at": None}

    def is_done():
        state["steps"] += 1
        ctx = decider_network.context
        if ctx.block_tower.is_complete:
            if state["done_at"] is None:
                state["done_at"] = state["steps"]
                print(f"[run] behavior 가 완료 보고 ({state['steps']} 스텝)", flush=True)
            # 탑이 물리적으로 안정될 시간을 조금 준다
            return state["steps"] > state["done_at"] + 120
        if state["steps"] % 2000 == 0:
            print(f"[run] {state['steps']} 스텝... 탑 높이={ctx.block_tower.height}", flush=True)
        return state["steps"] >= args.max_steps

    world.run(simulation_app, render=args.gui, loop_fast=True, play_on_entry=True, is_done_cb=is_done)

    elapsed = time.time() - state["t0"]
    completed = state["done_at"] is not None
    ok, report = evaluate_stack(cubes, tower_position)
    print(report)
    print(f"\n스텝: {state['steps']} (완료 보고 {state['done_at']}) / 소요 {elapsed:.1f}s")
    print(f"STACK_RESULT: {'PASS' if (ok and completed) else 'FAIL'}")


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
finally:
    simulation_app.close()
