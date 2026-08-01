"""[ROBOE] carry-high 이후 신규 증상 재현 진단 - "탑 근처에서 맴돌며 착지 못함".

diag16 트라이얼 2 (고정 재현): Yellow 를 (0.49,-0.37) 에서 집어 탑 (0.25,0.30) 에
올리는 도중 EE 가 (0.19, 0.46, 0.42) 부근에서 5cm 폭으로 진동하며 영구 미수렴.

무엇을 보나 (가설 판별용):
  A. 명령 목표(arm.target_prim) 실제 좌표 - carry-high 목표/접근점/최종점 중 무엇인가
  B. d_xy 와 히스테리시스 경계(0.16/0.20) 상호작용 - 경계에서 목표가 널뛰는가
  C. 관절 위치 vs 한계 - 손목/베이스가 한계에 붙어 자세를 못 만드는가
  D. 장애물 억제 상태 - 탑 상단(Red) 척력이 하강을 막는가

실행: python eval/diag_hover.py
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
CUBE_HALF = 0.0515 / 2.0

# diag16 트라이얼 2 스폰 (같은 시드의 3번째 샘플) - 실측 재현 배치
SPAWNS = {
    "RedCube": (0.61, 0.35),
    "BlueCube": (0.42, -0.21),
    "YellowCube": (0.49, -0.37),
    "GreenCube": (0.47, 0.22),
}


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
    sample.tower_position = np.array([0.25, 0.30, 0.0])
    pump(sample.load_world_async())

    rng = np.random.default_rng(1)
    for name, obj in sample.cubes.items():
        x, y = SPAWNS[name]
        yaw = rng.uniform(0, np.pi / 2)
        obj.set_world_pose(position=np.array([x, y, CUBE_HALF]),
                           orientation=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        obj.set_linear_velocity(np.zeros(3))
        obj.set_angular_velocity(np.zeros(3))
    # 트라이얼과 동일: belief 오염 시작 (고스트를 엉뚱한 곳으로)
    for name, obj in sample.belief_cubes.items():
        obj.set_world_pose(position=np.array([0.45, -0.1, CUBE_HALF]))

    pump(sample.on_event_async())
    world = sample.get_world()
    ctx = sample.decider_network.context
    arm = sample.robot.arm

    art = sample.robot  # CortexFranka(Articulation)
    try:
        lo = art.dof_properties["lower"][:7]
        hi = art.dof_properties["upper"][:7]
    except Exception:
        lo = hi = None

    logging = False
    for step in range(5000):
        world.step(render=True)

        carrying_yellow = ctx.in_gripper is not None and ctx.in_gripper.name == "YellowCube"
        if not logging and carrying_yellow:
            logging = True
            print(f"\n### Yellow 파지 @step {step} - 고밀도 로깅 시작 ###", flush=True)
        if logging and step % 60 == 0:
            eff = np.asarray(arm.get_fk_p())
            try:
                tgt = np.asarray(arm.target_prim.get_world_pose()[0])
            except Exception:
                tgt = None
            pt = ctx.placement_target_eff_T
            pt_p = np.round(pt[:3, 3], 3) if pt is not None else None
            d_xy = (np.linalg.norm(pt[:2, 3] - eff[:2]) if pt is not None else None)
            q = np.asarray(art.get_joint_positions())[:7]
            margins = None
            if lo is not None:
                margins = np.round(np.minimum(q - lo, hi - q), 2)  # 관절한계까지 여유(rad)
            sup = {n: (not b.collision_avoidance_enabled) for n, b in ctx.blocks.items()}
            stack = " > ".join(str(e) for e in sample.decider_network._decider_state.stack)
            print(f"[{step:5d}] EE={np.round(eff,3)} 명령목표={np.round(tgt,3) if tgt is not None else '-'} "
                  f"배치목표={pt_p} d_xy={d_xy:.3f}" if d_xy is not None else
                  f"[{step:5d}] EE={np.round(eff,3)} 명령목표={tgt} 배치목표=None", flush=True)
            print(f"        관절한계여유={margins} 억제됨={[n for n,v in sup.items() if v]}", flush=True)
            print(f"        스택={stack[:120]}", flush=True)

        if ctx.block_tower.height >= 2 and logging:
            print(f"\n### Yellow 배치 성공 @step {step} - 증상 미재현 ###", flush=True)
            break

    print("HOVER_DIAG: DONE")


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
finally:
    sys.stdout.flush()
    simulation_app.close()
