"""[ROBOE] zero-shot 후보 2: Grounding DINO tiny (open-vocabulary 검출의 사실상 표준).

원리: DETR 계열 검출기(DINO)에 BERT 텍스트 인코더를 결합. 문장 프롬프트를
매 프레임 함께 인코딩해 '언어 표현 <-> 이미지 영역' 대응(grounding)을 푼다.
YOLO-World보다 언어 이해가 유연하지만, 텍스트를 매번 처리하므로 ~10배 느리다.

라벨 매핑 주의: 반환 라벨이 매칭된 "문구 조각"(예: "red cube", "cube")이라
색 키워드로 클래스에 사상하고, 색이 없는 조각은 버린다.

실행: training/.venv/bin/python eval/zeroshot/run_gdino.py
  (transformers 필요 - eval/zeroshot/requirements-extra.txt)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import evaluate, load_val, save_report  # noqa: E402

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # noqa: E402

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
TEXT = "a red cube. a yellow cube. a green cube. a blue cube."
KEYWORD2CLS = [("red", 0), ("yellow", 1), ("green", 2), ("blue", 3)]

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to("cuda").eval()


def to_cls(label):
    label = label.lower()
    for kw, c in KEYWORD2CLS:
        if kw in label:
            return c
    return None  # 색 없는 조각("cube" 단독 등)은 클래스 판정 불가 -> 제외


samples = load_val()
preds, lat = {}, []
for s in samples:
    im = Image.open(s["img"]).convert("RGB")
    t0 = time.perf_counter()
    inputs = processor(images=im, text=TEXT, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs)
    try:  # transformers 버전에 따라 인자명이 다르다 (threshold <-> box_threshold)
        res = processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=0.10, text_threshold=0.20,
            target_sizes=[im.size[::-1]])[0]
    except TypeError:
        res = processor.post_process_grounded_object_detection(
            out, inputs.input_ids, box_threshold=0.10, text_threshold=0.20,
            target_sizes=[im.size[::-1]])[0]
    lat.append(time.perf_counter() - t0)
    labels = res.get("text_labels", res.get("labels"))
    items = []
    for score, label, box in zip(res["scores"], labels, res["boxes"]):
        c = to_cls(str(label))
        if c is not None:
            items.append((c, float(score), [float(v) for v in box]))
    preds[s["img"].name] = items

rep = evaluate(preds, samples, "Grounding DINO (tiny)", TEXT, lat,
               notes="fp32. threshold=0.10으로 곡선 채점, 운영점 0.25. "
                     "색 키워드 없는 매칭 조각은 제외.")
save_report("gdino_tiny", rep)
