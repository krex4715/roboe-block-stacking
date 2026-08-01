"""[ROBOE] Qwen2.5-VL 워커 - 생성형 VLM 이 박스를 '문장으로 생성'하는 검출 서버.

gdino_worker 와 같은 프로토콜(JSON 라인, 비동기 우편함의 서버 쪽)이라 detector_hub 의
_WorkerClient 를 그대로 쓴다. 차이는 성격뿐이다:
- 챗봇형 모델에게 프롬프트로 지시하고 좌표를 토큰으로 받아 파싱한다 (그라운딩)
- 장당 초 단위로 느리다 -> 우편함 구조 덕에 시뮬은 안 막히고, 보정이 ~0.2Hz 로 올 뿐
- **신뢰도가 없다**: 생성 출력엔 score 개념이 없어 0.99 고정으로 내보낸다.
  게이트가 무의미해지므로 보호는 bridge 의 작업공간/일관성/동결 장치가 전담한다.

좌표 복원: Qwen 은 스마트 리사이즈된 입력 해상도 기준 절대좌표를 출력한다.
image_grid_thw(패치 격자) x 패치 14px 로 입력 해상도를 복원해 원본으로 스케일.

단독 테스트:
    echo '{"cmd":"quit"}' | training/.venv/bin/python \
        isaac_ext/roboe_block_stacking/perception/qwen_worker.py
"""

import json
import re
import sys
import time

import torch
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLM
except ImportError:  # 신버전 transformers 의 통합 클래스
    from transformers import AutoModelForImageTextToText as VLM

from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = (
    "Detect every small colored cube in this image. There are up to four cubes: "
    "red, yellow, green (light green), and blue. "
    'Output ONLY a JSON array like [{"bbox_2d": [x1, y1, x2, y2], "label": "red cube"}] '
    "with one entry per cube. No other text."
)
KEYWORDS = [("red", "red_cube", 0), ("yellow", "yellow_cube", 1),
            ("green", "green_cube", 2), ("blue", "blue_cube", 3)]
FIXED_SCORE = 0.99  # 생성형 출력엔 신뢰도가 없다 (모듈 주석 참고)
ORIG_W, ORIG_H = 1280, 720  # ZED 캡처 해상도


def main():
    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = VLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()
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
            t = time.perf_counter()
            messages = [{"role": "user", "content": [
                {"type": "image", "image": f"data:image/jpeg;base64,{req['jpg_b64']}"},
                {"type": "text", "text": PROMPT}]}]
            text = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            decoded = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                             skip_special_tokens=True)[0]

            grid = inputs["image_grid_thw"][0].tolist()  # [t, h_patch, w_patch]
            sx, sy = ORIG_W / (grid[2] * 14), ORIG_H / (grid[1] * 14)

            dets = []
            m = re.search(r"\[.*\]", decoded, re.DOTALL)
            if m:
                try:
                    for d in json.loads(m.group(0)):
                        box = d.get("bbox_2d") or d.get("bbox")
                        label = str(d.get("label", "")).lower()
                        if not box or len(box) != 4:
                            continue
                        for kw, cname, cid in KEYWORDS:
                            if kw in label:
                                dets.append({"class": cname, "class_id": cid,
                                             "score": FIXED_SCORE,
                                             "box": [round(box[0] * sx, 1), round(box[1] * sy, 1),
                                                     round(box[2] * sx, 1), round(box[3] * sy, 1)]})
                                break
                except json.JSONDecodeError:
                    pass  # 형식 붕괴 프레임 = 검출 0건 (생성형의 실패 모드)
            print(json.dumps({"detections": dets,
                              "ms": round((time.perf_counter() - t) * 1000, 1)}), flush=True)
        except Exception as exc:  # 한 프레임의 실패가 워커를 죽이면 안 된다
            print(json.dumps({"detections": [], "ms": 0.0, "error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
