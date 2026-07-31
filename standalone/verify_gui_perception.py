"""[ROBOE] GUI 예제의 인식 루프가 실제로 도는지 헤드리스로 검증.

GUI 에서 바로 돌려보면 예외가 async 태스크에 삼켜져 원인이 안 보인다.
그래서 같은 코드 경로(LOAD -> Start -> physics 콜백)를 헤드리스로 먼저 태운다.

실행:
    python standalone/verify_gui_perception.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import asyncio  # noqa: E402

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.examples.browser")
enable_extension("isaacsim.examples.interactive")
for _ in range(5):
    simulation_app.update()

from isaacsim.examples.interactive.user_examples.roboe_block_stacking.roboe_stacking_example import (  # noqa: E402
    RoboeBlockStacking,
)

BEHAVIOR_PATH = str(REPO / "isaac_ext" / "roboe_block_stacking" / "behavior" / "block_stacking_behavior.py")


def pump(coro, label, limit=4000):
    fut = asyncio.ensure_future(coro)
    for _ in range(limit):
        if fut.done():
            break
        simulation_app.update()
    if not fut.done():
        raise TimeoutError(label)
    fut.result()
    print(f"    [ok] {label}")


def main():
    messages = []
    sample = RoboeBlockStacking(perception_fn=lambda t: messages.append(t))
    sample.behavior = BEHAVIOR_PATH

    print("\n[1] LOAD (씬 구성 + ZED-X + 검출기 로드)")
    pump(sample.load_world_async(), "load_world_async")
    print(f"    검출기: {'로드됨' if sample.detector is not None else '없음'}")
    if messages:
        print("    UI 메시지:\n      " + messages[-1].replace("\n", "\n      "))
    if sample.detector is None:
        print("GUI_PERCEPTION: FAIL - 검출기가 로드되지 않음 (models/best.torchscript 확인)")
        return

    print("\n[2] Start (physics 콜백 등록 + 재생)")
    pump(sample.on_event_async(), "on_event_async")

    print("\n[3] 시뮬레이션 스텝 - 인식 루프가 도는지 확인")
    world = sample.get_world()
    for i in range(240):  # 4초분
        world.step(render=True)
    print(f"    인식 호출 횟수(스텝 카운터): {sample._perception_step}")
    print(f"    마지막 추정: {len(sample.last_estimates)}개 클래스")
    for cls, p in sorted(sample.last_estimates.items()):
        print(f"        {cls:12s} ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")
    if messages:
        print("    UI 패널 내용:\n      " + messages[-1].replace("\n", "\n      "))

    print("\n[4] 인식 OFF 토글")
    sample.set_perception_enabled(False)
    for _ in range(60):
        world.step(render=True)
    print(f"    OFF 후 UI: {messages[-1][:60]}")

    ok = len(sample.last_estimates) == 4
    print(f"\nGUI_PERCEPTION: {'PASS' if ok else 'FAIL'} (기준: 큐브 4개 모두 추정)")


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
    print("\nGUI_PERCEPTION: FAIL (예외)")
    sys.stdout.flush()
    sys.stderr.flush()
finally:
    simulation_app.close()
