"""[ROBOE] zero-shot 검출 대안 비교 - 공통 채점기.

목적: "RGB -> 클래스+박스" 생산자 자리를 놓고 파인튜닝 YOLO(SDG 학습)와
zero-shot 후보(YOLO-World / Grounding DINO / Qwen2.5-VL)를 **같은 잣대**로 비교한다.
- 문제지: data/cubes val 300장 + Replicator GT (라벨링 추가 비용 0)
- 채점: 클래스별 AP50(VOC 방식) + 운영점(conf 0.25) 정밀도/재현율/혼동행렬
- 모든 러너가 이 모듈 하나로 채점 -> 모델 간 채점 방식 차이가 끼어들 수 없다.

주의: 이 val 세트는 조명 랜덤화가 걸려 있어, "색 이름"으로 판별하는
zero-shot 계열의 최대 약점(노랑<->연두)을 정확히 스트레스 테스트한다.
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "cubes"
RESULTS = Path(__file__).resolve().parent / "results"

CLASSES = ["red_cube", "yellow_cube", "green_cube", "blue_cube"]
NC = len(CLASSES)
W, H = 1280, 720

CONF_OPERATING = 0.25  # 운영점: 실제 브리지 게이트와 같은 급의 컷
IOU_THR = 0.50


def load_val(stride=1):
    """val 이미지 + GT 로드. GT: YOLO 포맷(cls cx cy w h, 정규화) -> 픽셀 xyxy."""
    samples = []
    for p in sorted((DATA / "images" / "val").glob("*.jpg"))[::stride]:
        gt = {}
        lbl = DATA / "labels" / "val" / (p.stem + ".txt")
        if lbl.exists():
            for line in lbl.read_text().strip().splitlines():
                c, cx, cy, w, h = line.split()
                c = int(c)
                cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                gt.setdefault(c, []).append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        samples.append({"img": p, "gt": gt})
    return samples


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _ap50(cls_id, preds_by_img, samples):
    """VOC 방식 AP@IoU0.5 (신뢰도 내림차순 탐욕 매칭 + PR 곡선 아래 면적)."""
    entries, gts, npos = [], [], 0
    for i, s in enumerate(samples):
        boxes = s["gt"].get(cls_id, [])
        gts.append([{"box": b, "used": False} for b in boxes])
        npos += len(boxes)
        for c, conf, box in preds_by_img.get(s["img"].name, []):
            if c == cls_id:
                entries.append((conf, i, box))
    if npos == 0:
        return float("nan")
    entries.sort(key=lambda e: -e[0])
    tp = np.zeros(len(entries))
    fp = np.zeros(len(entries))
    for k, (_, i, box) in enumerate(entries):
        best, bj = 0.0, -1
        for j, g in enumerate(gts[i]):
            v = iou(box, g["box"])
            if v > best:
                best, bj = v, j
        if best >= IOU_THR and not gts[i][bj]["used"]:
            gts[i][bj]["used"] = True
            tp[k] = 1
        else:
            fp[k] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / npos
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _operating(preds_by_img, samples):
    """운영점(conf>=0.25) 지표: 기하 매칭(클래스 무관, IoU>=0.5) 후 클래스 대조.
    -> 혼동행렬[GT, 예측], 배경 오검(FP), 미검(FN)을 분리해 볼 수 있다."""
    conf_mat = np.zeros((NC, NC), dtype=int)
    fp_bg = np.zeros(NC, dtype=int)   # GT 어디에도 안 붙은 예측 (배경 오검/중복검출)
    fn = np.zeros(NC, dtype=int)      # 아무 예측도 안 붙은 GT (미검)
    for s in samples:
        preds = sorted((p for p in preds_by_img.get(s["img"].name, []) if p[1] >= CONF_OPERATING),
                       key=lambda p: -p[1])
        gt_list = [(c, b) for c, boxes in s["gt"].items() for b in boxes]
        used = [False] * len(gt_list)
        for pc, _, pbox in preds:
            best, bj = 0.0, -1
            for j, (gc, gbox) in enumerate(gt_list):
                if used[j]:
                    continue
                v = iou(pbox, gbox)
                if v > best:
                    best, bj = v, j
            if best >= IOU_THR:
                used[bj] = True
                conf_mat[gt_list[bj][0], pc] += 1
            else:
                fp_bg[pc] += 1
        for j, u in enumerate(used):
            if not u:
                fn[gt_list[j][0]] += 1
    prec, rec = {}, {}
    for c in range(NC):
        pred_c = int(conf_mat[:, c].sum() + fp_bg[c])
        gt_c = int(conf_mat[c, :].sum() + fn[c])
        prec[CLASSES[c]] = round(conf_mat[c, c] / pred_c, 4) if pred_c else None
        rec[CLASSES[c]] = round(conf_mat[c, c] / gt_c, 4) if gt_c else None
    return {
        "conf_thr": CONF_OPERATING,
        "precision": prec,
        "recall": rec,
        "confusion_gt_x_pred": conf_mat.tolist(),
        "fp_background": {CLASSES[c]: int(fp_bg[c]) for c in range(NC)},
        "missed": {CLASSES[c]: int(fn[c]) for c in range(NC)},
    }


def _policy_pick(preds_by_img, samples):
    """브리지 정책 시뮬레이션: 클래스별 최고 신뢰도 1개만 픽(순수 argmax, 게이트 무시).

    실제 cortex_bridge는 '클래스별 최고 score 1개 + 게이트'만 belief로 발행하므로,
    이 픽이 진짜 그 색 큐브 위에 있는 비율(pick_acc)이 스태킹 성공을 가장 직접
    예측한다. wrong = 픽이 엉뚱한 곳(다른 큐브/배경)에 붙음 -> belief 오염 위험.
    게이트를 무시하는 이유: zero-shot 계열은 신뢰도 보정이 안 돼 있어(실측)
    고정 게이트로 비교하면 순위 매기기 능력 자체를 볼 수 없기 때문."""
    stats = {c: {"correct": 0, "wrong": 0, "none": 0, "n_gt": 0} for c in range(NC)}
    for s in samples:
        preds = preds_by_img.get(s["img"].name, [])
        for c in range(NC):
            gts = s["gt"].get(c, [])
            if not gts:
                continue
            stats[c]["n_gt"] += 1
            cand = [p for p in preds if p[0] == c]
            if not cand:
                stats[c]["none"] += 1
                continue
            best = max(cand, key=lambda p: p[1])
            if max(iou(best[2], g) for g in gts) >= IOU_THR:
                stats[c]["correct"] += 1
            else:
                stats[c]["wrong"] += 1
    out = {}
    for c, v in stats.items():
        out[CLASSES[c]] = {
            "pick_acc": round(v["correct"] / v["n_gt"], 4) if v["n_gt"] else None,
            "wrong_pick": v["wrong"], "no_pick": v["none"], "frames_with_gt": v["n_gt"],
        }
    accs = [v["pick_acc"] for v in out.values() if v["pick_acc"] is not None]
    out["mean_pick_acc"] = round(float(np.mean(accs)), 4) if accs else None
    return out


def evaluate(preds_by_img, samples, model_name, prompts, latency_s, notes=""):
    """공통 리포트 생성. latency_s: 이미지별 벽시계 시간(전처리+추론+후처리 전부)."""
    ap = {CLASSES[c]: round(_ap50(c, preds_by_img, samples), 4) for c in range(NC)}
    lat = np.array(latency_s[3:] if len(latency_s) > 6 else latency_s)  # 워밍업 3장 제외
    report = {
        "model": model_name,
        "prompts": prompts,
        "n_images": len(samples),
        "mAP50": round(float(np.nanmean(list(ap.values()))), 4),
        "ap50_per_class": ap,
        "operating": _operating(preds_by_img, samples),
        "policy_pick": _policy_pick(preds_by_img, samples),
        "latency_ms": {"mean": round(float(lat.mean()) * 1000, 1),
                       "median": round(float(np.median(lat)) * 1000, 1)},
        "notes": notes,
    }
    return report


def save_report(name, report):
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[zeroshot] {name}: mAP50={report['mAP50']} "
          f"lat={report['latency_ms']['mean']}ms -> {out.relative_to(REPO)}")
    return out
