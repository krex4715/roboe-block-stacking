"""[ROBOE] YOLO TorchScript 추론기 (isaacsim 환경의 순정 torch 로만 동작).

**의존성 설계**: 이 파일은 torch / torchvision / numpy / cv2 만 쓴다. ultralytics 를 import 하지
않으므로 isaacsim 환경에 새 패키지를 하나도 설치하지 않는다. 대신 YOLOv8 의 원시 출력
텐서를 직접 디코드해야 하는데, 그 디코드가 ultralytics 와 정말 같은 결과를 내는지는
`training/verify_torchscript_decode.py` 가 학습 venv 에서 증명한다 (배포 전 게이트).

**YOLOv8 원시 출력 형식** (NMS 미적용):
    (1, 4 + nc, N)   N = 8400 @ imgsz 640
    앞 4채널 = cx, cy, w, h  (letterbox 입력 픽셀 좌표계)
    뒤 nc채널 = 클래스별 점수 (이미 sigmoid 적용됨, objectness 없음 - v5 와 다른 점)

**전처리 규약** (ultralytics 와 반드시 동일해야 함):
    letterbox(비율 유지 축소 + 114 회색 패딩) -> RGB -> /255 -> NCHW float32
"""

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

PAD_VALUE = 114  # ultralytics 기본 letterbox 패딩 값


def letterbox(image, size):
    """비율을 유지한 채 size(정사각)에 맞추고 남는 곳을 회색으로 채운다.

    Returns:
        padded: (size, size, 3) uint8
        ratio: 원본 -> 축소 비율
        (dw, dh): 좌/상 패딩 픽셀
    """
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )
    return padded, ratio, (left, top)


def decode_predictions(raw, conf_threshold, iou_threshold, ratio, pad, orig_shape, max_det=64):
    """YOLOv8 원시 출력 -> 원본 이미지 좌표계의 (boxes, scores, classes).

    ultralytics 의 non_max_suppression 을 대체하는 부분. torchvision.ops.batched_nms 를 쓰면
    클래스별로 독립 NMS 가 되어 서로 다른 색 큐브가 겹쳐 있어도 하나만 남지 않는다.
    """
    if raw.ndim == 3:
        raw = raw[0]
    if raw.shape[0] < raw.shape[1]:  # (4+nc, N) -> (N, 4+nc)
        raw = raw.transpose(0, 1)

    boxes_xywh = raw[:, :4]
    class_scores = raw[:, 4:]
    scores, classes = class_scores.max(dim=1)

    keep = scores > conf_threshold
    if not torch.any(keep):
        empty = torch.zeros((0, 4), device=raw.device)
        return empty, scores[:0], classes[:0]
    boxes_xywh, scores, classes = boxes_xywh[keep], scores[keep], classes[keep]

    # xywh -> xyxy (letterbox 좌표계)
    xy, wh = boxes_xywh[:, :2], boxes_xywh[:, 2:4]
    boxes = torch.cat([xy - wh / 2, xy + wh / 2], dim=1)

    idx = batched_nms(boxes, scores, classes, iou_threshold)[:max_det]
    boxes, scores, classes = boxes[idx], scores[idx], classes[idx]

    # letterbox 좌표 -> 원본 좌표 (패딩 제거 후 비율 복원)
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes /= ratio
    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h - 1)
    return boxes, scores, classes


class CubeDetector:
    """TorchScript YOLO 로 큐브를 검출한다.

    사용:
        det = CubeDetector("models/best.torchscript", ["red_cube", ...])
        dets = det(rgb)   # [{"class": "red_cube", "score": .., "box": (x0,y0,x1,y1)}, ...]
    """

    def __init__(self, model_path, class_names, imgsz=640, conf=0.5, iou=0.5,
                 device=None, half=False):
        self.class_names = list(class_names)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.half = bool(half) and self.device.type == "cuda"

        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()
        if self.half:
            self.model.half()
        self._warmup()
        self.last_latency_ms = 0.0

    def _warmup(self):
        """첫 추론은 CUDA 커널 컴파일 때문에 수십~수백 ms 걸린다.
        시뮬레이션 루프 안에서 그 지연이 튀지 않도록 미리 한 번 돌려둔다."""
        dummy = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device,
                            dtype=torch.half if self.half else torch.float32)
        with torch.inference_mode():
            self.model(dummy)

    def preprocess(self, rgb):
        padded, ratio, pad = letterbox(rgb, self.imgsz)
        tensor = torch.from_numpy(np.ascontiguousarray(padded)).to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.half() if self.half else tensor.float()
        return tensor / 255.0, ratio, pad

    @torch.inference_mode()
    def __call__(self, rgb):
        start = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        if start:
            start.record()

        tensor, ratio, pad = self.preprocess(rgb)
        raw = self.model(tensor)
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        boxes, scores, classes = decode_predictions(
            raw.float(), self.conf, self.iou, ratio, pad, rgb.shape
        )

        if start:
            end.record()
            torch.cuda.synchronize()
            self.last_latency_ms = start.elapsed_time(end)

        out = []
        for box, score, cls in zip(boxes.cpu().numpy(), scores.cpu().numpy(), classes.cpu().numpy()):
            idx = int(cls)
            out.append({
                "class": self.class_names[idx] if idx < len(self.class_names) else str(idx),
                "class_id": idx,
                "score": float(score),
                "box": tuple(float(v) for v in box),
            })
        return out

    def best_per_class(self, detections):
        """클래스별 최고 점수 1개만 남긴다. 큐브는 색마다 하나뿐이라는 사전지식을 쓰는 것."""
        best = {}
        for d in detections:
            if d["class"] not in best or d["score"] > best[d["class"]]["score"]:
                best[d["class"]] = d
        return best
