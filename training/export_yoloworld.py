"""[ROBOE] YOLO-World v2 를 zero-shot 그대로 TorchScript 로 굽는다 (GUI 인식 소스용).

핵심 아이디어: YOLO-World 는 프롬프트를 CLIP 텍스트 인코더로 임베딩해 검출 헤드에
주입한다. set_classes() 시점에 그 임베딩이 모델에 **고정(bake)**되므로, 이후 export 하면
텍스트 분기가 사라진 "일반 YOLOv8 모양(1, 4+nc, N)"의 그래프가 나온다.
-> 런타임(isaacsim 환경)은 기존 `perception/detector.py` 의 TorchScript 경로를 그대로 쓴다.
   **학습 0 + 런타임 신규 패키지 0** 이 동시에 성립하는 유일한 zero-shot 후보.

주의 (오프라인 실측 근거, eval/zeroshot):
- imgsz=1280 필수: 큐브가 ~38px 라 640 추론에선 신뢰도가 붕괴한다 (mAP50 0.39 -> 0.68)
- conf 게이트 0.05: zero-shot 신뢰도는 보정이 안 돼 있어 기존 0.5 게이트면 전멸한다.
  게이트 값은 meta 로 내보내고 런타임이 읽는다 ("게이트는 모델의 보정 특성을 따라간다")
- 프롬프트 "light green cube": green 대비 mAP50 0.525 -> 0.680 (프롬프트 민감도 실측)

배포 전 게이트: verify_torchscript_decode.py 와 같은 방법 - ultralytics 로 같은 TorchScript
를 돌린 결과와 내 디코드를 비교 (전처리 동일 -> 남는 차이는 디코드뿐).

실행: training/.venv/bin/python training/export_yoloworld.py
"""

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLOWorld  # noqa: E402
from ultralytics import YOLO  # noqa: E402

sys.path.insert(0, str(REPO / "isaac_ext" / "roboe_block_stacking"))
from perception.detector import CubeDetector  # noqa: E402  (런타임과 같은 코드)

PROMPTS = ["red cube", "yellow cube", "light green cube", "blue cube"]
CLASS_NAMES = ["red_cube", "yellow_cube", "green_cube", "blue_cube"]  # 프롬프트와 같은 순서
IMGSZ = 1280
# 게이트 0.003 의 근거 (val 100장 실측): TP argmax-pick 신뢰도 p25=0.0045, 중앙값 0.0105.
# TP/오검 분포가 겹쳐 게이트로는 분리 불가(0.005 게이트: TP 72% vs 오검 55% 생존) ->
# 게이트는 거의 열고, 보호는 bridge 의 작업공간/일관성 스냅/동결 안전장치가 맡는다.
CONF = 0.003
OUT_TS = REPO / "models" / "yoloworld_v2s.torchscript"
OUT_META = REPO / "models" / "yoloworld_meta.json"


def iou_xyxy(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    weights = Path(__file__).parent / "yolov8s-worldv2.pt"
    model = YOLOWorld(str(weights) if weights.exists() else "yolov8s-worldv2.pt")
    model.set_classes(PROMPTS)
    # device=0 명시: 기본(cpu) export 는 CUDA_VISIBLE_DEVICES 를 지워버려
    # 같은 프로세스의 후속 parity 검증에서 CUDA 가 사라진다 (실측).
    exported = Path(model.export(format="torchscript", imgsz=IMGSZ, device=0))
    OUT_TS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, OUT_TS)
    print(f"[export] {exported.name} -> {OUT_TS.relative_to(REPO)} "
          f"({OUT_TS.stat().st_size / 1e6:.1f} MB)")

    # ---- parity: ultralytics(같은 TS) vs 내 디코드 ----
    ref = YOLO(str(OUT_TS))
    mine = CubeDetector(str(OUT_TS), CLASS_NAMES, imgsz=IMGSZ, conf=CONF, iou=0.5)

    images = sorted((REPO / "data" / "cubes" / "images" / "val").glob("*.jpg"))[:8]
    total_ref = total_mine = matched = cls_mismatch = 0
    ious = []
    for path in images:
        bgr = cv2.imread(str(path))
        r = ref.predict(source=bgr, imgsz=IMGSZ, conf=CONF, iou=0.5, verbose=False)[0]
        ref_dets = [(int(c), tuple(map(float, b)))
                    for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy())]
        my_dets = mine(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        total_ref += len(ref_dets)
        total_mine += len(my_dets)
        used = set()
        for rc, rb in ref_dets:
            best_i, best_iou = -1, 0.0
            for i, md in enumerate(my_dets):
                if i in used:
                    continue
                v = iou_xyxy(rb, md["box"])
                if v > best_iou:
                    best_i, best_iou = i, v
            if best_i >= 0 and best_iou >= 0.99:
                used.add(best_i)
                matched += 1
                ious.append(best_iou)
                cls_mismatch += int(my_dets[best_i]["class_id"] != rc)
        print(f"  {path.name}: ref {len(ref_dets)} / mine {len(my_dets)}")

    OUT_META.write_text(json.dumps({
        "class_names": CLASS_NAMES,
        "prompts": PROMPTS,
        "imgsz": IMGSZ,
        "conf": CONF,
        "bridge_min_score": CONF,
        "source": "yolov8s-worldv2 + set_classes (zero-shot - 학습 0, 어휘만 지정)",
        "offline_eval": {"mAP50": 0.6799, "mean_pick_acc": 0.6971,
                         "note": "eval/zeroshot/results/yoloworld_lightgreen.json"},
    }, indent=2, ensure_ascii=False))
    print(f"[export] meta -> {OUT_META.relative_to(REPO)}")

    if ious:
        print(f"parity: 박스 {total_ref}/{total_mine}/매칭 {matched}, "
              f"IoU 평균 {np.mean(ious):.5f} 최소 {np.min(ious):.5f}, 클래스 불일치 {cls_mismatch}")
    ok = total_ref > 0 and total_ref == total_mine == matched and cls_mismatch == 0
    print(f"EXPORT_YOLOWORLD: {'PASS' if ok else 'FAIL'}")


main()
