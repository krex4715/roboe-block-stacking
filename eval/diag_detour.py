"""[ROBOE] GUI 증상 재현 진단 - "연두 집을 때 노랑 지점으로 갔다가 온다".

기본 스폰(GUI 와 동일) 시나리오에서, 노랑 배치 이후~연두 파지 구간의
의사결정을 고해상도로 기록한다:
  - 결정 스택 (build-up 인가 teardown 인가!)
  - active_block 과 chosen_grasp 목표 좌표 (로봇이 실제로 향하는 곳)
  - 탑 구성과 순서 판정 (teardown 트리거 여부)
  - Yellow/Green 고스트·est·판정 사유

실행: python eval/diag_detour.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

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

BEHAVIOR = str(REPO / "isaac_ext/roboe_block_stacking/behavior/block_stacking_behavior.py")


def pump(coro, limit=6000):
    fut = asyncio.ensure_future(coro)
    for _ in range(limit):
        if fut.done():
            break
        simulation_app.update()
    fut.result()


def main():
    sample = RoboeBlockStacking(perception_fn=lambda t: None)
    sample.behavior = BEHAVIOR
    sample.belief_mode = "ghost"
    pump(sample.load_world_async())
    pump(sample.on_event_async())
    world = sample.get_world()
    ctx = sample.decider_network.context

    logging = False
    for step in range(9000):
        world.step(render=True)

        # 노랑이 탑에 오르면(높이 2) 고해상도 로깅 시작, 연두가 잡히면 종료
        if not logging and ctx.block_tower.height >= 2:
            logging = True
            print(f"\n### 탑 높이 2 도달 @step {step} - 고해상도 로깅 시작 ###", flush=True)
        if logging and step % 20 == 0:
            stack = " > ".join(str(e) for e in sample.decider_network._decider_state.stack)
            ab = ctx.active_block
            grasp_p = None
            if ab is not None and ab.chosen_grasp is not None:
                grasp_p = np.round(ab.chosen_grasp[:3, 3], 3)
            tower = [b.name for b in ctx.block_tower.stack]
            try:
                ee = np.round(np.asarray(sample.robot.arm.get_fk_p()), 3)
            except Exception:
                ee = None
            print(f"[{step:5d}] 스택={stack[:110]}", flush=True)
            print(f"        active={ab.name if ab else None} 파지목표={grasp_p} EE={ee} "
                  f"탑={tower} 순서OK={ctx.block_tower.current_stack_in_correct_order} "
                  f"in_gripper={ctx.in_gripper.name if ctx.in_gripper else None}", flush=True)
            for n in ("YellowCube", "GreenCube"):
                g = np.round(np.asarray(sample.belief_cubes[n].get_world_pose()[0]), 3)
                r = np.round(np.asarray(sample.cubes[n].get_world_pose()[0]), 3)
                cls = "yellow_cube" if n == "YellowCube" else "green_cube"
                est = sample.last_estimates.get(cls)
                print(f"        {n:10s} ghost={g} real={r} est={np.round(est,3) if est is not None else '-'} "
                      f"| {sample.bridge.last_reasons.get(n, '-')}", flush=True)

        if logging and sample.bridge.stats["grasp_events"] >= 3:
            print(f"\n### 연두 파지 성립 @step {step} - 로깅 종료 ###", flush=True)
            break
        if ctx.block_tower.is_complete:
            break

    print("DETOUR_DIAG: DONE")


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
finally:
    sys.stdout.flush()
    simulation_app.close()
