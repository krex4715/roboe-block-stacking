"""[ROBOE] 회귀 테스트 - 다른 예제를 거쳤다 돌아와도 LOAD 가 정상 동작하는가.

재현하는 버그:
    CortexBase.load_world_async 는 `if CortexWorld.instance() is None:` 일 때만 setup_scene()
    을 부른다. 그런데 월드 싱글톤은 SimulationContext._instance 하나를 모든 하위 클래스가
    공유하므로, 다른 예제를 로드했다 돌아오면 인스턴스가 살아남아 setup_scene() 이 건너뛰어진다.
    스테이지는 이미 비워졌으므로 robot/큐브/카메라 참조가 죽은 프림을 가리키고,
    setup_post_load() 에서 예외 -> GUI 에는 "Task exception was never retrieved" 로만 보인다.

이 스크립트는 GUI 없이 그 흐름(LOAD -> 다른 예제 -> LOAD)을 그대로 재현한다.

실행:
    python standalone/verify_example_reload.py
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-fix",
    action="store_true",
    help="우리 방어 코드를 우회하고 CortexBase 원본 경로로 로드 (버그 재현 확인용)",
)
cli_args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import asyncio  # noqa: E402

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.examples.browser")
enable_extension("isaacsim.examples.interactive")
for _ in range(5):
    simulation_app.update()

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import create_new_stage_async  # noqa: E402
from isaacsim.examples.interactive.cortex.cortex_base import CortexBase  # noqa: E402
from isaacsim.examples.interactive.user_examples.roboe_block_stacking.roboe_stacking_example import (  # noqa: E402
    RoboeBlockStacking,
)

BEHAVIOR_PATH = str(REPO / "isaac_ext" / "roboe_block_stacking" / "behavior" / "block_stacking_behavior.py")


def run_until_done(coro, label, timeout_updates=4000):
    """Kit 업데이트 루프를 돌리면서 코루틴이 끝날 때까지 기다린다.

    표준 run_until_complete 는 안 된다 - 코루틴 안에서 next_update_async() 를 기다리는데
    그 업데이트를 돌려줄 주체가 없어 교착된다. 그래서 직접 펌프한다.
    """
    fut = asyncio.ensure_future(coro)
    for _ in range(timeout_updates):
        if fut.done():
            break
        simulation_app.update()
    if not fut.done():
        raise TimeoutError(f"{label}: 시간 초과")
    fut.result()  # 예외가 있었다면 여기서 터진다 (GUI 라면 조용히 삼켜졌을 것)
    print(f"    [ok] {label}")


def check_scene(sample, label):
    """씬이 실제로 구성됐는지 확인 (setup_scene 이 건너뛰어지면 여기서 걸린다)."""
    world = sample.get_world()
    problems = []
    if sample.robot is None:
        problems.append("robot 없음")
    if sample.zed is None or sample.zed.camera is None:
        problems.append("ZED-X 없음")
    else:
        n_obs = len(sample.robot.registered_obstacles) if sample.robot else 0
        if n_obs != 4:
            problems.append(f"등록된 큐브 {n_obs}개 (기대 4개)")
    if sample.decider_network is None:
        problems.append("decider_network 없음")
    else:
        stack = sample.decider_network.context.block_tower.desired_stack
        if stack != ["RedCube", "YellowCube", "GreenCube", "BlueCube"]:
            problems.append(f"쌓기 순서 이상: {stack}")

    if problems:
        print(f"    [FAIL] {label}: " + ", ".join(problems))
        return False
    print(f"    [ok] {label}: 로봇/큐브4/카메라/decider 모두 정상")
    return True


def main():
    sample = RoboeBlockStacking()
    sample.behavior = BEHAVIOR_PATH
    results = []

    # --no-fix: 우리 오버라이드를 건너뛰고 CortexBase 원본을 직접 호출한다.
    # 이 모드에서 FAIL 이 나야 테스트가 실제로 버그를 잡고 있다는 뜻이다.
    def do_load():
        if cli_args.no_fix:
            return CortexBase.load_world_async(sample)
        return sample.load_world_async()

    if cli_args.no_fix:
        print("\n*** --no-fix 모드: 방어 코드를 우회합니다 (버그 재현 기대) ***")

    print("\n[1회차] ROBOE 예제 LOAD")
    run_until_done(do_load(), "load_world_async #1")
    results.append(check_scene(sample, "1회차 씬"))

    print("\n[중간] 다른 예제를 연 상황 재현 (새 스테이지 + 일반 World)")
    run_until_done(create_new_stage_async(), "create_new_stage_async")
    other_world = World()  # 다른 예제가 하는 일. 싱글톤이 남는 것이 버그의 핵심
    print(f"    다른 예제의 world 타입: {type(other_world).__name__} "
          f"(싱글톤 공유 여부: {other_world is World.instance()})")

    print("\n[2회차] ROBOE 예제 다시 LOAD  <- 여기서 터지던 지점")
    run_until_done(do_load(), "load_world_async #2")
    results.append(check_scene(sample, "2회차 씬"))

    print("\n[3회차] 한 번 더 (연속 LOAD 안정성)")
    run_until_done(do_load(), "load_world_async #3")
    results.append(check_scene(sample, "3회차 씬"))

    print(f"\nRELOAD_TEST: {'PASS' if all(results) else 'FAIL'}")


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
    print("\nRELOAD_TEST: FAIL (예외)")
    sys.stdout.flush()
    sys.stderr.flush()
finally:
    simulation_app.close()
