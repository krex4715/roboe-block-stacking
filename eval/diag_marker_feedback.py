"""[ROBOE] 결정적 실험 - debug_draw 마커가 ZED 카메라에 보이고 YOLO 에 검출되는가.

가설(스크린샷 근거): 뷰포트용 인식 마커(클래스 색 점)가 카메라 render product 에도
렌더링되어, YOLO 가 자기 마커를 큐브로 오검출하는 **자기 관측 피드백 루프**가 존재한다.

실험: 큐브가 없는 빈 위치에 노란 점(마커와 동일 스펙)만 그리고
  1) ZED 이미지의 해당 픽셀이 노랗게 찍히는지
  2) YOLO 가 그 자리에서 yellow_cube 를 검출하는지
확인한다.

실행: python eval/diag_marker_feedback.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import json  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402

sys.path.insert(0, str(REPO / "isaac_ext" / "roboe_block_stacking"))
from perception.detector import CubeDetector  # noqa: E402
from perception.zed_camera import ZedXCamera  # noqa: E402
from scene_setup import add_lighting  # noqa: E402


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_lighting()  # 큐브는 일부러 안 넣는다 - 마커만 검출되는지 보려고

    zed = ZedXCamera()
    zed.spawn()
    world.reset()
    zed.initialize()

    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()

    # 런타임과 동일 스펙의 마커: 큰 점(20) + 작은 점(8), 빈 위치 (0.45, -0.1)
    P_BIG = (0.45, -0.10, 0.115)   # ghost 마커 위치 (큐브가 있다면 +0.09)
    P_SMALL = (0.45, -0.10, 0.075)

    meta = json.loads((REPO / "models" / "model_meta.json").read_text())
    det = CubeDetector(str(REPO / "models" / "best.torchscript"),
                       meta["class_names"], imgsz=meta["imgsz"], conf=0.3)

    CASES = [
        ("클래스색(노랑) - 기존 방식", (1.0, 1.0, 0.2, 1.0), (1.0, 1.0, 0.2, 1.0)),
        ("중립색(흰/회백) - 수정안", (1.0, 1.0, 1.0, 1.0), (0.6, 0.6, 0.6, 1.0)),
    ]

    for case_name, c_big, c_small in CASES:
        rgb = None
        for i in range(60):
            draw.clear_points()
            draw.draw_points([P_BIG, P_SMALL], [c_big, c_small], [20, 8])
            world.step(render=True)
            rgb, _ = zed.capture()
            if rgb is not None and i > 40:
                break
        if rgb is None:
            print(f"[{case_name}] 캡처 실패")
            continue

        uv = np.asarray(zed.camera.get_image_coords_from_world_points(
            np.array([P_BIG], dtype=float)))[0]
        u, v = int(round(uv[0])), int(round(uv[1]))
        patch = rgb[max(0, v - 6):v + 7, max(0, u - 6):u + 7].reshape(-1, 3).astype(float)
        mean_rgb = np.round(patch.mean(axis=0)).astype(int)

        detections = det(rgb)
        hit = any(abs((d["box"][0] + d["box"][2]) / 2 - u) < 40
                  and abs((d["box"][1] + d["box"][3]) / 2 - v) < 40 for d in detections)
        det_s = ", ".join(f"{d['class']} {d['score']:.2f}" for d in detections) or "없음"
        print(f"\n=== {case_name} ===")
        print(f"  마커 픽셀({u},{v}) 색 = {mean_rgb} / 빈 씬 YOLO 검출: {det_s}")
        print(f"  판정: {'★ 큐브로 오검출됨 (오염!)' if hit else '오검출 없음 (안전)'}")

    print(f"\nMARKER_FEEDBACK: DONE")


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
finally:
    sys.stdout.flush()
    simulation_app.close()
