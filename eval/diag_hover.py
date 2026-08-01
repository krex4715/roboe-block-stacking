"""[ROBOE] hover 실패 정밀 재현 - diag 시드42 트라이얼2 의 스폰+yaw 완전 복원판.

계측으로 확정된 사건 순서 (diag17):
  step ~240  탑 위 정상 하강 중 (d_xy=0.015, Red 척력 억제)
  step ~600  돌연 하강 중단, 큐브 든 채 대선회
  step ~720+ 관절 0,1(베이스/어깨)이 한계 여유 0.04/0.06 rad 로 감긴 채 영구 hover

이 스크립트가 판별할 것: 중단 직전 배치 목표(위치+자세)가 무엇으로 바뀌었나.
  가설 H1: 하강 중인 Yellow 가 탑 멤버로 오인 -> 목표가 'Yellow 위' 로 점프
  가설 H2: 목표 orientation 90도 플립 -> 손목 재배향 요구 -> 후퇴 선회
주의: diag_trials 와 달리 belief 오염 없음 (그쪽도 안 한다) - 조건 동일화.

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

# 시드 42 트라이얼 2 의 rng 스트림 완전 복원값 (스폰 xy + yaw)
SPAWNS = {
    "RedCube": (0.6077, 0.3498, np.radians(63.02)),
    "BlueCube": (0.4234, -0.2055, np.radians(28.11)),
    "YellowCube": (0.4866, -0.3738, np.radians(74.90)),
    "GreenCube": (0.4723, 0.2244, np.radians(72.43)),
}


def pump(coro, limit=6000):
    fut = asyncio.ensure_future(coro)
    for _ in range(limit):
        if fut.done():
            break
        simulation_app.update()
    fut.result()


def yaw_of(T):
    """eff 목표 T 의 접근 자세 yaw (x축 기준, deg)."""
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


def main():
    sample = RoboeBlockStacking(perception_fn=lambda t: None)
    sample.behavior = BEHAVIOR
    sample.belief_mode = "ghost"
    sample.tower_position = np.array([0.25, 0.30, 0.0])
    pump(sample.load_world_async())

    for name, obj in sample.cubes.items():
        x, y, yaw = SPAWNS[name]
        obj.set_world_pose(position=np.array([x, y, CUBE_HALF]),
                           orientation=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        obj.set_linear_velocity(np.zeros(3))
        obj.set_angular_velocity(np.zeros(3))

    pump(sample.on_event_async())
    world = sample.get_world()
    ctx = sample.decider_network.context
    arm = sample.robot.arm
    try:
        lo = sample.robot.dof_properties["lower"][:7]
        hi = sample.robot.dof_properties["upper"][:7]
    except Exception:
        lo = hi = None

    logging = False
    for step in range(4500):
        world.step(render=True)

        grip = getattr(ctx, "in_gripper", None)
        if not logging and grip is not None and grip.name == "YellowCube":
            logging = True
            print(f"\n### Yellow 파지 @step {step} ###", flush=True)
        if logging and step % 20 == 0:
            eff = np.asarray(arm.get_fk_p())
            pt = ctx.placement_target_eff_T
            tower = [b.name for b in ctx.block_tower.stack]
            if pt is not None:
                d_xy = np.linalg.norm(pt[:2, 3] - eff[:2])
                pt_s = f"p={np.round(pt[:3,3],3)} yaw={yaw_of(pt):6.1f}"
            else:
                d_xy, pt_s = -1, "None"
            q = np.asarray(sample.robot.get_joint_positions())[:7]
            marg = np.round(np.minimum(q - lo, hi - q), 2) if lo is not None else None
            print(f"[{step:5d}] EE={np.round(eff,3)} 목표[{pt_s}] d_xy={d_xy:.3f} "
                  f"탑={tower} grip={grip.name if grip else None}", flush=True)
            print(f"        한계여유={marg} | R:{sample.bridge.last_reasons.get('RedCube','-')}"
                  f" | Y:{sample.bridge.last_reasons.get('YellowCube','-')}", flush=True)

        if logging and ctx.block_tower.height >= 2 and getattr(ctx, "in_gripper", None) is None:
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
