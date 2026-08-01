"""[ROBOE] M6 실패 진단 - run_trials 와 같은 흐름 + 상세 계측 (같은 seed 로 재현).

run_trials.py 가 성공률만 주는 반면, 이 스크립트는 실패의 '왜'를 보여준다:
매 1500스텝마다 결정스택 + 고스트/실제/추정 좌표 + bridge 통계를 덤프한다.

실행:
    python eval/diag_trials.py --trials 3
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max-steps", type=int, default=6000)
args = parser.parse_args()

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
from isaacsim.examples.interactive.user_examples.roboe_block_stacking.scene_setup import (  # noqa: E402
    CUBE_HALF,
    CUBE_SIZE,
)

BEHAVIOR = str(REPO / "isaac_ext/roboe_block_stacking/behavior/block_stacking_behavior.py")
SCATTER_X, SCATTER_Y = (0.28, 0.72), (-0.55, 0.38)
MIN_GAP = CUBE_SIZE * 1.6
C2P = {"red_cube": "RedCube", "yellow_cube": "YellowCube",
       "green_cube": "GreenCube", "blue_cube": "BlueCube"}


def pump(coro, label, limit=6000):
    fut = asyncio.ensure_future(coro)
    for _ in range(limit):
        if fut.done():
            break
        simulation_app.update()
    fut.result()


def sample_spawns(rng, tower_xy):
    pts = []
    for _ in range(2000):
        if len(pts) == 4:
            break
        p = np.array([rng.uniform(*SCATTER_X), rng.uniform(*SCATTER_Y)])
        r = np.linalg.norm(p)
        # 파지 유효 작업공간: r >= 0.40 (실측 근거 - r~0.335 파지 시 팔이 접힌
        # 자세로 관절한계에 갇혀 RMPFlow 가 이탈 불가. diag14 트라이얼 6/10 재현.
        # 스톡 예제도 r 0.5~0.8 에서만 동작하도록 설계됨. 실제 셀의 작업 반경 명세와 동일 개념)
        if not (0.40 <= r <= 0.75):
            continue
        if np.linalg.norm(p - tower_xy) <= 0.16:
            continue
        if any(np.linalg.norm(p - q) <= MIN_GAP for q in pts):
            continue
        pts.append(p)
    return pts


def main():
    tower_xy = np.array([0.25, 0.30])
    rng = np.random.default_rng(args.seed)
    sample = RoboeBlockStacking(perception_fn=lambda t: None)
    sample.behavior = BEHAVIOR
    sample.belief_mode = "ghost"
    sample.tower_position = np.array([0.25, 0.30, 0.0])

    n_pass = 0
    for trial in range(args.trials):
        spawns = sample_spawns(rng, tower_xy)
        pump(sample.load_world_async(), "LOAD")
        for (name, obj), p in zip(sample.cubes.items(), spawns):
            yaw = rng.uniform(0, np.pi / 2)
            obj.set_world_pose(position=np.array([p[0], p[1], CUBE_HALF]),
                               orientation=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        print(f"\n===== trial {trial} =====")
        print("스폰(물리):", {n: np.round(np.asarray(o.get_world_pose()[0])[:2], 2).tolist()
                             for n, o in sample.cubes.items()}, flush=True)

        pump(sample.on_event_async(), "Start")
        world = sample.get_world()
        ctx = sample.decider_network.context

        done_at = None
        try:
            _lo = sample.robot.dof_properties["lower"][:7]
            _hi = sample.robot.dof_properties["upper"][:7]
        except Exception:
            _lo = _hi = None
        for step in range(args.max_steps):
            world.step(render=True)
            # [진단] 배치 이동(carry) 중 고밀도 계측 - hover 미수렴 원인 규명용.
            #   명령목표(arm.target_prim) / d_xy(carry-high 히스테리시스 입력) /
            #   관절한계 여유 / 척력 억제 목록
            if step % 120 == 0 and getattr(ctx, "in_gripper", None) is not None \
                    and ctx.placement_target_eff_T is not None:
                try:
                    arm = sample.robot.arm
                    ee = np.asarray(arm.get_fk_p())
                    cmd = np.asarray(arm.target_prim.get_world_pose()[0])
                    pt = ctx.placement_target_eff_T
                    d_xy = np.linalg.norm(pt[:2, 3] - ee[:2])
                    q = np.asarray(sample.robot.get_joint_positions())[:7]
                    marg = (np.round(np.minimum(q - _lo, _hi - q), 2)
                            if _lo is not None else None)
                    sup = [n for n, b in ctx.blocks.items() if not b.collision_avoidance_enabled]
                    print(f"      [carry {step:5d}] EE={np.round(ee,3)} 명령목표={np.round(cmd,3)} "
                          f"d_xy={d_xy:.3f} 한계여유={marg} 억제={sup}", flush=True)
                except Exception as exc:
                    print(f"      [carry {step:5d}] 계측 실패: {exc}", flush=True)
            if step % 1500 == 0 and step:
                stack = " > ".join(str(e) for e in sample.decider_network._decider_state.stack)
                st = sample.bridge.stats
                # 배치 정체(go_target(False)) 원인 규명용: EE vs 목표, 장애물 상태
                try:
                    ee = np.asarray(sample.robot.arm.get_fk_p())
                    tgt = ctx.placement_target_eff_T
                    tgt_s = np.round(tgt[:3, 3], 3) if tgt is not None else None
                    obs = {n: b.collision_avoidance_enabled for n, b in ctx.blocks.items()}
                    grip = getattr(ctx, "in_gripper", None)
                    print(f"      EE={np.round(ee,3)} 배치목표={tgt_s} "
                          f"in_gripper={grip.name if grip else None} 장애물on={[n for n,v in obs.items() if v]}",
                          flush=True)
                except Exception as exc:
                    print(f"      (계측 실패: {exc})", flush=True)
                print(f"  [step {step:4d}] 탑={ctx.block_tower.height} 파지={st['grasp_events']} "
                      f"발행={st['published']} 스냅={st['snapped']} 재획득={st['reacquired']} "
                      f"동결={st['frozen']}\n      스택: {stack[:150]}", flush=True)
                for n in sample.cubes:
                    g = np.asarray(sample.belief_cubes[n].get_world_pose()[0])
                    r = np.asarray(sample.cubes[n].get_world_pose()[0])
                    reason = sample.bridge.last_reasons.get(n, "-")
                    print(f"      {n:11s} ghost={np.round(g,2)} real={np.round(r,2)} "
                          f"오차={np.linalg.norm(g-r)*1000:5.0f}mm | {reason}", flush=True)
            if ctx.block_tower.is_complete:
                if done_at is None:
                    done_at = step
                    print(f"  완료 보고 @{step}", flush=True)
                if step > done_at + 150:
                    break

        rows = [(n, np.asarray(o.get_world_pose()[0], dtype=float)) for n, o in sample.cubes.items()]
        stacked = sorted(rows, key=lambda r: r[1][2])
        order = [n for n, _ in stacked]
        offs = {n: round(float(np.linalg.norm(p[:2] - tower_xy)) * 1000) for n, p in rows}
        ok = (done_at is not None and order == ["RedCube", "YellowCube", "GreenCube", "BlueCube"]
              and max(offs.values()) <= 40)
        n_pass += int(ok)
        print(f"  최종: {'PASS' if ok else 'FAIL'} done={done_at} 순서={order}")
        print(f"  탑중심 xy 오프셋(mm): {offs}", flush=True)

    print(f"\nDIAG_RESULT: {n_pass}/{args.trials} PASS")


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
finally:
    sys.stdout.flush()
    simulation_app.close()
