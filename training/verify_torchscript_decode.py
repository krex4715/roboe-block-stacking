"""[ROBOE] 배포 전 게이트 - 내가 직접 짠 디코드가 ultralytics 와 같은 결과를 내는가.

**왜 이 검증이 필요한가**:
런타임(isaacsim 환경)에는 ultralytics 를 설치하지 않는다. 대신 TorchScript 원시 출력을
`perception/detector.py` 가 직접 디코드한다. 그런데 디코드는 조용히 틀리기 쉽다 —
채널 순서, xywh/xyxy, letterbox 좌표 복원, NMS 방식 중 하나만 어긋나도
"박스가 조금씩 밀린 채로 그럭저럭 도는" 상태가 되어 나중에 3D 오차로만 나타난다.

그래서 **학습 venv 에서** ultralytics 정답과 내 디코드를 같은 이미지로 비교한다.
이게 통과하기 전에는 Isaac Sim 연동으로 넘어가지 않는다.

실행:
    training/.venv/bin/python training/verify_torchscript_decode.py
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str, default=str(REPO / "models" / "best.pt"))
parser.add_argument("--torchscript", type=str, default=str(REPO / "models" / "best.torchscript"))
parser.add_argument("--images", type=str, default=str(REPO / "data" / "cubes" / "images" / "val"))
parser.add_argument("--n", type=int, default=8)
parser.add_argument("--conf", type=float, default=0.5)
parser.add_argument("--iou", type=float, default=0.5)
parser.add_argument("--min-iou", type=float, default=0.99, help="통과 기준 박스 IoU")
args = parser.parse_args()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# 런타임과 **같은 코드**를 import 한다 (복붙하면 나중에 갈라지므로)
sys.path.insert(0, str(REPO / "isaac_ext" / "roboe_block_stacking"))
from perception.detector import CubeDetector  # noqa: E402


def iou_xyxy(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    meta_path = REPO / "models" / "model_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    class_names = meta.get("class_names", ["red_cube", "yellow_cube", "green_cube", "blue_cube"])
    imgsz = int(meta.get("imgsz", 640))

    images = sorted(Path(args.images).glob("*.jpg"))[: args.n]
    if not images:
        print(f"VERIFY_DECODE: FAIL - 이미지 없음: {args.images}")
        return

    # 기준을 **TorchScript 모델**로 잡는다 (.pt 가 아니라).
    # ultralytics 는 .pt 추론에서 rect letterbox(32의 배수로만 패딩, 예: 640x384)를 쓰고
    # 내보낸 모델에는 고정 정사각 입력(640x640)을 쓴다. .pt 를 기준으로 삼으면 전처리 차이가
    # 섞여 들어와 "디코드가 맞는가"를 순수하게 볼 수 없다.
    # 같은 TorchScript 를 ultralytics 로도 돌리면 전처리가 동일해지므로,
    # 남는 차이는 오직 내 디코드/NMS/좌표복원 뿐이다.
    ref = YOLO(args.torchscript)
    mine = CubeDetector(args.torchscript, class_names, imgsz=imgsz,
                        conf=args.conf, iou=args.iou)
    ref_pt = YOLO(args.weights) if Path(args.weights).exists() else None

    total_ref = total_mine = matched = 0
    ious, cls_mismatch = [], 0
    pt_agree, pt_total = 0, 0

    for path in images:
        bgr = cv2.imread(str(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        r = ref.predict(source=bgr, imgsz=imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]
        ref_dets = [
            {"class": r.names[int(c)], "box": tuple(float(v) for v in b), "score": float(s)}
            for b, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                               r.boxes.cls.cpu().numpy())
        ]
        my_dets = mine(rgb)

        # 참고용: 원본 .pt 와도 개수/클래스가 맞는지 (전처리가 달라 IoU 는 조금 낮을 수 있음)
        if ref_pt is not None:
            rp = ref_pt.predict(source=bgr, imgsz=imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]
            pt_classes = sorted(rp.names[int(c)] for c in rp.boxes.cls.cpu().numpy())
            my_classes = sorted(d["class"] for d in my_dets)
            pt_total += 1
            pt_agree += int(pt_classes == my_classes)

        total_ref += len(ref_dets)
        total_mine += len(my_dets)

        # 기준 박스마다 가장 잘 맞는 내 박스를 찾는다
        used = set()
        per_img = []
        for rd in ref_dets:
            best_i, best_iou = -1, 0.0
            for i, md in enumerate(my_dets):
                if i in used:
                    continue
                v = iou_xyxy(rd["box"], md["box"])
                if v > best_iou:
                    best_i, best_iou = i, v
            if best_i >= 0 and best_iou >= args.min_iou:
                used.add(best_i)
                matched += 1
                ious.append(best_iou)
                if my_dets[best_i]["class"] != rd["class"]:
                    cls_mismatch += 1
                per_img.append(f"{rd['class']}:IoU {best_iou:.4f}")
            else:
                per_img.append(f"{rd['class']}:미매칭(최고 IoU {best_iou:.3f})")
        print(f"  {path.name}: ref {len(ref_dets)} / mine {len(my_dets)} | " + ", ".join(per_img))

    print(f"\n기준 박스 {total_ref} / 내 박스 {total_mine} / 매칭 {matched}")
    if ious:
        print(f"IoU 평균 {np.mean(ious):.5f} / 최소 {np.min(ious):.5f}")
    print(f"클래스 불일치: {cls_mismatch}")
    if pt_total:
        print(f"원본 .pt 와 클래스 집합 일치: {pt_agree}/{pt_total} 장 (참고용)")
    print(f"추론 지연(마지막 프레임): {mine.last_latency_ms:.2f} ms")

    ok = (total_ref > 0 and total_ref == total_mine == matched and cls_mismatch == 0)
    print(f"\nVERIFY_DECODE: {'PASS' if ok else 'FAIL'} "
          f"(기준: 박스 개수 일치 + 전부 IoU>={args.min_iou} + 클래스 일치)")


main()
