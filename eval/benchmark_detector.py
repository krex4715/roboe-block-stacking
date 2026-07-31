"""[ROBOE] 검출기 추론 지연 측정 (isaacsim 환경에서 실행).

**왜 따로 재는가**: 인식이 시뮬레이션 루프 안에서 돌기 때문에 프레임 예산을 얼마나 먹는지가
설계 제약이다. 60Hz 물리 스텝(16.7ms) 대비 얼마인지, 어느 단계가 비싼지를 알아야
발행 주기(Hz)를 정할 수 있다.

단계를 나눠 잰다:
    preprocess  letterbox(cv2, CPU) + H2D 복사 + 정규화
    inference   TorchScript forward
    decode      conf 필터 + NMS + 좌표 복원

주의: GPU가 다른 작업(예: Isaac Sim 렌더링)으로 포화돼 있으면 값이 몇 배로 튄다.
측정할 때는 다른 GPU 작업이 없어야 한다.

실행:
    python eval/benchmark_detector.py --n 200
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "isaac_ext" / "roboe_block_stacking"))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=str(REPO / "models" / "best.torchscript"))
parser.add_argument("--images", type=str, default=str(REPO / "data" / "cubes" / "images" / "val"))
parser.add_argument("--n", type=int, default=200)
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--half", action="store_true", help="fp16 로도 재본다")
args = parser.parse_args()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from perception.detector import CubeDetector, decode_predictions  # noqa: E402


def summarize(name, samples_ms):
    s = sorted(samples_ms)
    return {
        "stage": name,
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "p95": s[int(len(s) * 0.95) - 1],
        "max": s[-1],
    }


def main():
    meta_path = REPO / "models" / "model_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    class_names = meta.get("class_names", ["red_cube", "yellow_cube", "green_cube", "blue_cube"])
    imgsz = int(meta.get("imgsz", 640))

    paths = sorted(Path(args.images).glob("*.jpg"))
    if not paths:
        print(f"BENCH: FAIL - 이미지 없음: {args.images}")
        return
    images = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths[: min(50, len(paths))]]
    print(f"[bench] 이미지 {len(images)}장 순환, {args.n}회 측정 (워밍업 {args.warmup}회)")
    print(f"[bench] GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    for half in ([False, True] if args.half else [False]):
        det = CubeDetector(args.model, class_names, imgsz=imgsz, conf=0.5, iou=0.5, half=half)
        for i in range(args.warmup):
            det(images[i % len(images)])
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        pre_ms, inf_ms, dec_ms, total_ms, n_det = [], [], [], [], []
        for i in range(args.n):
            rgb = images[i % len(images)]
            t0 = time.perf_counter()
            tensor, ratio, pad = det.preprocess(rgb)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.perf_counter()
            with torch.inference_mode():
                raw = det.model(tensor)
                if isinstance(raw, (list, tuple)):
                    raw = raw[0]
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t2 = time.perf_counter()
            with torch.inference_mode():
                boxes, _, _ = decode_predictions(raw.float(), det.conf, det.iou, ratio, pad, rgb.shape)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t3 = time.perf_counter()

            pre_ms.append((t1 - t0) * 1e3)
            inf_ms.append((t2 - t1) * 1e3)
            dec_ms.append((t3 - t2) * 1e3)
            total_ms.append((t3 - t0) * 1e3)
            n_det.append(len(boxes))

        tag = "fp16" if half else "fp32"
        print(f"\n=== {tag} ===  평균 검출 {np.mean(n_det):.2f}개/프레임")
        print(f"{'단계':12s} {'평균':>9s} {'중앙':>9s} {'p95':>9s} {'최대':>9s}")
        for name, s in (("preprocess", pre_ms), ("inference", inf_ms),
                        ("decode", dec_ms), ("TOTAL", total_ms)):
            r = summarize(name, s)
            print(f"{name:12s} {r['mean']:8.2f}ms {r['median']:8.2f}ms "
                  f"{r['p95']:8.2f}ms {r['max']:8.2f}ms")
        med = statistics.median(total_ms)
        print(f"→ 60Hz 물리 스텝(16.7ms) 대비 {med/16.7*100:.1f}% / "
              f"이론 최대 {1000/med:.0f} FPS")


main()
