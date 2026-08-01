"""[ROBOE] zero-shot 후보 3: Qwen2.5-VL-3B-Instruct (생성형 VLM).

원리: 챗봇형 비전-언어 모델에게 "큐브 박스를 JSON으로 출력해"라고 지시하면
박스 좌표를 **문장 생성하듯** 출력한다(그라운딩 능력 내장). 가장 유연하지만
(추론·설명도 가능) 자막처럼 토큰을 하나씩 생성하므로 초 단위로 느리다.
-> 로봇 10Hz 루프용이 아니라 오프라인 분석/자동 라벨링용 성격. 느려서 서브셋만 채점.

좌표 복원 주의: Qwen은 스마트 리사이즈된 입력 해상도 기준 절대좌표를 출력한다.
image_grid_thw(패치 격자) x 패치크기(14)로 입력 해상도를 복원해 원본으로 스케일.

신뢰도 없음: 생성 출력엔 conf가 없어 1.0으로 둔다 -> AP보다 정밀/재현이 본질 지표.

실행: training/.venv/bin/python eval/zeroshot/run_qwen.py
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import evaluate, load_val, save_report  # noqa: E402

import torch  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLM
except ImportError:  # 신버전 transformers의 통합 클래스
    from transformers import AutoModelForImageTextToText as VLM

from qwen_vl_utils import process_vision_info  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SUBSET_STRIDE = 10  # 300장 중 30장 (장당 ~2s라 전수는 비경제적)
PROMPT = (
    "Detect every small colored cube in this image. There are up to four cubes: "
    "red, yellow, green (light green), and blue. "
    'Output ONLY a JSON array like [{"bbox_2d": [x1, y1, x2, y2], "label": "red cube"}] '
    "with one entry per cube. No other text."
)
KEYWORD2CLS = [("red", 0), ("yellow", 1), ("green", 2), ("blue", 3)]

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = VLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda").eval()

samples = load_val(stride=SUBSET_STRIDE)
preds, lat = {}, []
for s in samples:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(s["img"])},
        {"type": "text", "text": PROMPT}]}]
    t0 = time.perf_counter()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    decoded = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True)[0]
    lat.append(time.perf_counter() - t0)

    # 리사이즈 입력 해상도 -> 원본(1280x720) 스케일 복원
    grid = inputs["image_grid_thw"][0].tolist()  # [t, h_patch, w_patch]
    in_h, in_w = grid[1] * 14, grid[2] * 14
    sx, sy = 1280 / in_w, 720 / in_h

    items = []
    m = re.search(r"\[.*\]", decoded, re.DOTALL)
    if m:
        try:
            for d in json.loads(m.group(0)):
                box = d.get("bbox_2d") or d.get("bbox")
                label = str(d.get("label", "")).lower()
                if not box or len(box) != 4:
                    continue
                c = next((ci for kw, ci in KEYWORD2CLS if kw in label), None)
                if c is not None:
                    items.append((c, 1.0, [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]))
        except json.JSONDecodeError:
            pass  # 형식 붕괴 프레임 = 검출 0건으로 채점 (생성형의 실패 모드도 데이터)
    preds[s["img"].name] = items

rep = evaluate(preds, samples, "Qwen2.5-VL-3B (생성형 VLM)", PROMPT, lat,
               notes=f"서브셋 {len(samples)}장(stride {SUBSET_STRIDE}). bf16, greedy. "
                     "conf 미제공 -> 1.0 고정이므로 AP보다 정밀/재현이 본질 지표.")
save_report("qwen25vl_3b", rep)
