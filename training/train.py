"""[ROBOE] YOLO 파인튜닝 + 검증 + 배포용 export.

**학습 전용 venv 에서 실행한다** (isaacsim 환경 아님 - requirements.txt 주석 참고):
    training/.venv/bin/python training/train.py

산출물:
    models/best.pt           ultralytics 원본 가중치 (재학습/재export 용)
    models/best.torchscript  ★ 런타임(isaacsim 환경)이 실제로 로드하는 파일
    models/best.onnx         이식성 확인용 (현재 런타임에서 쓰지는 않음)
    models/model_meta.json   클래스 순서 · imgsz · 전처리 규약 (추론 코드와의 계약)

**왜 YOLOv8n 인가** (README/발표용 요약):
  - 4클래스 · 고정 시점 · 단순 배경 문제에 큰 모델은 과하다. n 모델로 충분하고 추론이 수 ms 다
  - 실시간성이 중요하다: 인식이 시뮬레이션 루프 안에서 돌기 때문에 프레임 예산을 먹으면
    시뮬이 느려지고 로봇 제어 주기에도 영향을 준다
  - 대안 비교: 색 임계처리(과제의 'AI 모델' 조건 미충족), torchvision Faster R-CNN
    (추가 의존성 0이지만 추론 ~10배 느림 - fallback 으로만 유지)
"""

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default=str(REPO / "data" / "cubes" / "dataset.yaml"))
parser.add_argument("--model", type=str, default="yolov8n.pt")
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--device", type=str, default="0")
parser.add_argument("--project", type=str, default=str(REPO / "training" / "runs"))
parser.add_argument("--name", type=str, default="cubes")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

from ultralytics import YOLO  # noqa: E402


def main():
    models_dir = REPO / "models"
    models_dir.mkdir(exist_ok=True)

    model = YOLO(args.model)
    print(f"[train] {args.model} -> {args.data} (imgsz={args.imgsz}, epochs={args.epochs})")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=args.seed,
        # 합성 데이터 자체에 이미 도메인 랜덤화(조명/자세/배치)가 들어있고,
        # 런타임 카메라가 고정이라 기하 증강은 약하게 준다.
        # 대신 색 관련 증강(hsv_h)은 노랑/연두 구분을 흔들 수 있어 보수적으로 둔다.
        hsv_h=0.010,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.05,
        scale=0.2,
        fliplr=0.0,   # 좌우 반전 금지: 카메라가 고정이라 거울상 장면은 런타임에 존재하지 않는다
        mosaic=0.5,
        plots=True,
    )

    # 검증 - held-out val(다른 시드로 생성)에서 정직한 수치를 뽑는다
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device,
                        project=args.project, name=f"{args.name}_val", exist_ok=True)

    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    per_class = {}
    try:
        for i, ap50 in enumerate(metrics.box.ap50):
            per_class[names[int(metrics.box.ap_class_index[i])]] = float(ap50)
    except Exception:
        pass

    print("\n[val] mAP50 = %.4f / mAP50-95 = %.4f" % (metrics.box.map50, metrics.box.map))
    for k, v in per_class.items():
        print(f"    {k:12s} AP50 = {v:.4f}")

    # 혼동행렬 - 노랑/연두 혼동 여부가 색 선정 결정([[10 환경]])의 최종 검증이다.
    # 그림 파일만 보면 발표에서 "몇 개나 틀렸나"에 답할 수 없으므로 숫자로 뽑는다.
    confusion = None
    try:
        cm = metrics.confusion_matrix.matrix  # (nc+1, nc+1), 열=정답 행=예측
        labels = [names[i] for i in sorted(names)] + ["background"]
        print("\n[혼동행렬] 행=예측 / 열=정답")
        print("            " + "".join(f"{l[:9]:>11s}" for l in labels))
        for r, row in enumerate(cm):
            print(f"{labels[r][:11]:11s} " + "".join(f"{int(v):>11d}" for v in row))
        confusion = cm.astype(int).tolist()

        yi = [i for i, n in enumerate(labels) if n == "yellow_cube"]
        gi = [i for i, n in enumerate(labels) if n == "green_cube"]
        if yi and gi:
            y, g = yi[0], gi[0]
            swap = int(cm[y][g]) + int(cm[g][y])
            print(f"\n노랑<->연두 상호 오분류: {swap}건 "
                  f"(연두를 노랑으로 {int(cm[y][g])}, 노랑을 연두로 {int(cm[g][y])})")
    except Exception as exc:
        print(f"[val] 혼동행렬 추출 실패: {exc}")

    # 발표/README 에 넣을 그림 복사
    media = REPO / "media" / "training"
    media.mkdir(parents=True, exist_ok=True)
    for run_dir in (Path(args.project) / args.name, Path(args.project) / f"{args.name}_val"):
        for png in run_dir.glob("*.png"):
            shutil.copy(png, media / f"{run_dir.name}_{png.name}")
    print(f"[val] 학습 그래프/혼동행렬 이미지 -> {media}")

    # 배포 산출물
    best = Path(model.trainer.best) if getattr(model, "trainer", None) else None
    if best and best.exists():
        shutil.copy(best, models_dir / "best.pt")
    export_model = YOLO(str(models_dir / "best.pt"))

    ts = export_model.export(format="torchscript", imgsz=args.imgsz, device=args.device)
    shutil.copy(ts, models_dir / "best.torchscript")
    try:
        onnx = export_model.export(format="onnx", imgsz=args.imgsz, opset=17, device=args.device)
        shutil.copy(onnx, models_dir / "best.onnx")
    except Exception as exc:
        print(f"[export] onnx 실패(무시): {exc}")

    # 추론 코드와의 계약. 클래스 순서가 어긋나면 조용히 색이 뒤바뀌므로 파일로 못 박는다.
    meta = {
        "class_names": [names[i] for i in sorted(names)],
        "imgsz": args.imgsz,
        "preprocess": {
            "letterbox": True, "pad_value": 114, "scale": "1/255", "channel_order": "RGB",
            "layout": "NCHW",
        },
        "output": "(1, 4+nc, N) - xywh(픽셀, letterbox 좌표계) + 클래스 점수, NMS 미적용",
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class_AP50": per_class,
        "confusion_matrix": confusion,
        "torch_version_note": "isaacsim 환경(torch 2.7.0)에서 로드하려고 학습 venv 도 2.7.0 으로 고정",
    }
    (models_dir / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n[export] models/ 에 저장 완료: best.pt / best.torchscript / model_meta.json")
    print(f"TRAIN_RESULT: {'PASS' if metrics.box.map50 >= 0.95 else 'CHECK'} (mAP50={metrics.box.map50:.4f})")


main()
