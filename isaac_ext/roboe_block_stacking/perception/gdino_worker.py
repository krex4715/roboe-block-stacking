"""[ROBOE] Grounding DINO 워커 - 학습 venv(transformers)에서 도는 zero-shot 검출 서버.

**왜 서브프로세스인가**: "isaacsim 환경에 신규 패키지 0" 원칙(README §2.3)을 지키면서
zero-shot 검출기를 라이브로 쓰기 위해, transformers 가 이미 있는 학습 venv 를 워커로
빌린다. 시뮬레이터 쪽(detector_hub)은 표준 라이브러리 + cv2 만으로 통신한다.

프로토콜 (stdin/stdout, 한 줄 = JSON 하나):
    시작 완료  ->  {"ready": true, "device": "...", "load_s": ...}
    요청       <-  {"jpg_b64": "<base64 JPEG>"}
    응답       ->  {"detections": [{"class","class_id","score","box"}...], "ms": ...}
    종료       <-  {"cmd": "quit"}

단독 테스트:
    echo '{"cmd":"quit"}' | training/.venv/bin/python \
        isaac_ext/roboe_block_stacking/perception/gdino_worker.py
"""

import base64
import io
import json
import sys
import time

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
TEXT = "a red cube. a yellow cube. a green cube. a blue cube."
# (키워드, 런타임 클래스명, 클래스 id) - 매칭 문구 조각에서 색 키워드로 클래스를 정한다
KEYWORDS = [("red", "red_cube", 0), ("yellow", "yellow_cube", 1),
            ("green", "green_cube", 2), ("blue", "blue_cube", 3)]
BOX_THRESHOLD = 0.10   # 오프라인 채점과 동일 (운영 게이트는 bridge min_score 가 담당)
TEXT_THRESHOLD = 0.20


def main():
    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device).eval()
    print(json.dumps({"ready": True, "device": device,
                      "load_s": round(time.perf_counter() - t0, 1)}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        if req.get("cmd") == "quit":
            break
        try:
            img = Image.open(io.BytesIO(base64.b64decode(req["jpg_b64"]))).convert("RGB")
            t = time.perf_counter()
            inputs = processor(images=img, text=TEXT, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            try:  # transformers 버전에 따라 인자명이 다르다
                res = processor.post_process_grounded_object_detection(
                    out, inputs.input_ids, threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD, target_sizes=[img.size[::-1]])[0]
            except TypeError:
                res = processor.post_process_grounded_object_detection(
                    out, inputs.input_ids, box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD, target_sizes=[img.size[::-1]])[0]
            labels = res.get("text_labels", res.get("labels"))
            dets = []
            for score, label, box in zip(res["scores"], labels, res["boxes"]):
                label = str(label).lower()
                for kw, cname, cid in KEYWORDS:
                    if kw in label:
                        dets.append({"class": cname, "class_id": cid,
                                     "score": round(float(score), 4),
                                     "box": [round(float(v), 1) for v in box]})
                        break
            print(json.dumps({"detections": dets,
                              "ms": round((time.perf_counter() - t) * 1000, 1)}), flush=True)
        except Exception as exc:  # 한 프레임의 실패가 워커를 죽이면 안 된다
            print(json.dumps({"detections": [], "ms": 0.0, "error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
