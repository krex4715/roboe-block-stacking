"""[ROBOE] 디코드 좌표 수학 단위 검증 (학습 모델 없이, torch 만으로).

`verify_torchscript_decode.py` 는 학습된 모델이 있어야 돌지만, 이 테스트는 그 전에
**좌표 변환만 따로** 검증한다. letterbox 패딩 제거와 비율 복원이 틀리면 박스가
일정하게 밀리는데, 그건 나중에 3D 위치 오차로만 나타나 추적이 어렵기 때문이다.

방법: 원본 좌표를 알고 있는 가짜 박스를 letterbox 좌표계로 **정방향** 변환해
가짜 원시 출력을 만들고, 디코더가 원래 좌표를 되돌려내는지 본다 (왕복 테스트).

실행 (어느 환경에서든):
    python eval/test_decode_math.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "isaac_ext" / "roboe_block_stacking"))

import numpy as np
import torch

from perception.detector import decode_predictions, letterbox

IMG_W, IMG_H = 1280, 720
IMGSZ = 640
NUM_CLASSES = 4


def main():
    rng = np.random.default_rng(0)

    # 실제 전처리와 같은 함수로 비율/패딩을 얻는다
    dummy = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    padded, ratio, pad = letterbox(dummy, IMGSZ)
    print(f"letterbox: {IMG_W}x{IMG_H} -> {padded.shape[1]}x{padded.shape[0]}, "
          f"ratio={ratio:.4f}, pad={pad}")
    assert padded.shape[0] == padded.shape[1] == IMGSZ, "정사각 입력이어야 한다"

    # 원본 좌표계의 정답 박스 (다양한 위치/크기/클래스)
    truth = []
    for i in range(6):
        w = rng.uniform(30, 120)
        h = rng.uniform(30, 120)
        cx = rng.uniform(w / 2 + 5, IMG_W - w / 2 - 5)
        cy = rng.uniform(h / 2 + 5, IMG_H - h / 2 - 5)
        truth.append((cx, cy, w, h, int(rng.integers(0, NUM_CLASSES))))

    # 원본 -> letterbox 정방향 변환 (디코더가 하는 일의 역)
    n_anchors = 32
    raw = torch.zeros((1, 4 + NUM_CLASSES, n_anchors))
    for i, (cx, cy, w, h, cls) in enumerate(truth):
        raw[0, 0, i] = cx * ratio + pad[0]
        raw[0, 1, i] = cy * ratio + pad[1]
        raw[0, 2, i] = w * ratio
        raw[0, 3, i] = h * ratio
        raw[0, 4 + cls, i] = 0.9
    # 나머지 앵커는 낮은 점수 -> conf 필터에서 걸러져야 한다
    raw[0, 4:, len(truth):] = 0.05

    boxes, scores, classes = decode_predictions(
        raw, conf_threshold=0.5, iou_threshold=0.5, ratio=ratio, pad=pad,
        orig_shape=(IMG_H, IMG_W, 3),
    )

    print(f"\n입력 정답 {len(truth)}개 -> 디코드 결과 {len(boxes)}개")
    if len(boxes) != len(truth):
        print(f"DECODE_MATH: FAIL - 개수 불일치 (저점수 앵커가 안 걸러졌거나 NMS 가 삼킴)")
        return

    # 점수순 정렬돼 나오므로 위치로 매칭
    max_err, cls_err = 0.0, 0
    for cx, cy, w, h, cls in truth:
        want = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        d = [np.abs(np.asarray(b) - want).max() for b in boxes.numpy()]
        j = int(np.argmin(d))
        max_err = max(max_err, d[j])
        if int(classes[j]) != cls:
            cls_err += 1
        print(f"    정답 xyxy={np.round(want,1)}  복원={np.round(boxes[j].numpy(),1)}  "
              f"오차={d[j]:.3f}px  클래스 {cls}->{int(classes[j])}")

    print(f"\n최대 좌표 오차: {max_err:.4f} px / 클래스 오류: {cls_err}")
    # 허용 1px: letterbox 의 resize 반올림(round) 때문에 서브픽셀 오차가 남는다
    ok = max_err < 1.0 and cls_err == 0
    print(f"DECODE_MATH: {'PASS' if ok else 'FAIL'} (기준: 좌표 오차 < 1px, 클래스 오류 0)")


main()
