"""[ROBOE] 기준선: 파인튜닝 YOLOv8n(SDG 학습)을 zero-shot 후보와 같은 채점기로 재채점.

학습 시 val 수치(mAP50 0.9949)를 그대로 인용하지 않고 동일 하네스로 다시 재는 이유:
채점 구현 차이(매칭 방식, 임계값)가 비교표에 끼어들 여지를 없애기 위함이다.
추론 해상도는 학습과 같은 640 (신뢰도 보정이 그 조건에서 이뤄졌으므로).

실행: cd training && .venv/bin/python ../eval/zeroshot/run_finetuned.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, evaluate, load_val, save_report  # noqa: E402

from ultralytics import YOLO  # noqa: E402

model = YOLO(str(REPO / "models" / "best.pt"))

samples = load_val()
preds, lat = {}, []
for s in samples:
    t0 = time.perf_counter()
    r = model.predict(str(s["img"]), conf=0.001, iou=0.7, imgsz=640, verbose=False)[0]
    lat.append(time.perf_counter() - t0)
    preds[s["img"].name] = [
        (int(c), float(cf), list(map(float, b)))
        for c, cf, b in zip(r.boxes.cls.cpu().numpy(),
                            r.boxes.conf.cpu().numpy(),
                            r.boxes.xyxy.cpu().numpy())
    ]
rep = evaluate(preds, samples, "YOLOv8n 파인튜닝 (SDG 2800장)", "(폐쇄셋 - 프롬프트 없음)", lat,
               notes="기준선. 학습 해상도 640으로 추론. 학습기 자체 val 수치는 0.9949였음.")
save_report("finetuned_yolov8n", rep)
