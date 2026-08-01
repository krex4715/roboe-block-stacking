"""[ROBOE] zero-shot 후보 1: YOLO-World v2 (실시간 open-vocabulary 검출기).

원리: CLIP 텍스트 인코더가 프롬프트("red cube")를 임베딩으로 바꿔 검출 헤드에
주입 -> **학습 0**, 어휘 지정만으로 해당 문구의 물체를 찾는다. 텍스트 임베딩은
set_classes() 시점에 한 번만 계산되므로 실행 속도는 일반 YOLO급(실전 투입 후보).

프롬프트 민감도도 함께 채점한다(연두를 "green"으로 부를 때 vs "light green"으로
부를 때) - zero-shot에선 프롬프트가 곧 하이퍼파라미터임을 보이는 실험.

실행 (ultralytics가 있는 학습 venv, 가중치가 training/에 떨어지도록 cwd 고정):
  cd training && .venv/bin/python ../eval/zeroshot/run_yoloworld.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import evaluate, load_val, save_report  # noqa: E402

from ultralytics import YOLO  # noqa: E402

PROMPT_SETS = {
    "green": ["red cube", "yellow cube", "green cube", "blue cube"],
    "lightgreen": ["red cube", "yellow cube", "light green cube", "blue cube"],
}

samples = load_val()
for tag, prompts in PROMPT_SETS.items():
    model = YOLO("yolov8s-worldv2.pt")  # 최초 1회 자동 다운로드 (cwd=training/)
    model.set_classes(prompts)
    preds, lat = {}, []
    for s in samples:
        t0 = time.perf_counter()
        # imgsz=1280: 큐브가 ~38px로 작아 기본 640 추론에선 신뢰도가 붕괴한다
        # (첫 채점에서 실측 - conf>=0.25 검출 0건). 원본 해상도로 공정 비교.
        r = model.predict(str(s["img"]), conf=0.001, iou=0.7, imgsz=1280, verbose=False)[0]
        lat.append(time.perf_counter() - t0)
        preds[s["img"].name] = [
            (int(c), float(cf), list(map(float, b)))
            for c, cf, b in zip(r.boxes.cls.cpu().numpy(),
                                r.boxes.conf.cpu().numpy(),
                                r.boxes.xyxy.cpu().numpy())
        ]
    rep = evaluate(preds, samples, "YOLO-World-v2 (s)", prompts, lat,
                   notes="학습 0. conf=0.001로 전체 곡선 채점, 운영점은 0.25.")
    save_report(f"yoloworld_{tag}", rep)
