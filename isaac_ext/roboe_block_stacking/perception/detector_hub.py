"""[ROBOE] 인식 소스 허브 - "RGB -> 클래스+박스" 생산자를 런타임에 갈아끼운다.

zero-shot 오프라인 비교(eval/zeroshot, README §2.5)를 GUI 라이브 데모로 연장하는 장치.
파이프라인에서 검출기는 생산자 자리 하나이므로(3D 추정·bridge·behavior 무수정)
여기서만 교체하면 나머지는 그대로 동작한다.

설계 원칙:
1. **검증된 기본 경로 무변경**: finetuned 백엔드는 기존과 동일한 인자로 CubeDetector 를
   만든다. zero-shot 백엔드는 사용자가 명시적으로 선택했을 때만 로드된다.
2. **isaacsim 환경 신규 패키지 0 유지**:
   - yoloworld: CLIP 텍스트 임베딩을 TorchScript 에 bake (training/export_yoloworld.py)
     -> 기존 CubeDetector 경로 그대로 재사용
   - gdino: transformers 가 있는 **학습 venv 를 서브프로세스 워커**로 빌린다
     (JSON 라인 프로토콜, perception/gdino_worker.py)
3. **느린 백엔드가 시뮬 루프를 막지 않는다**: 워커는 비동기 우편함(mailbox) 방식 -
   놀고 있을 때만 프레임을 제출하고, 반환은 항상 '가장 최근 완성본'이다.
   ghost belief 구조가 저속/스테일 인식을 견딘다는 설계 주장을 실제로 시험하는 장치.
4. **게이트는 모델의 보정 특성을 따라간다**: zero-shot 신뢰도는 보정이 안 돼 있어
   (실측: YOLO-World TP 중앙값 0.0105) 고정 게이트 0.5 를 쓰면 전멸한다.
   백엔드별 detector conf 와 bridge min_score 를 meta 에서 읽어 함께 전환한다.
"""

import base64
import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import carb
import cv2

from .detector import CubeDetector

_DEFAULT_CLASSES = ["red_cube", "yellow_cube", "green_cube", "blue_cube"]

# 백엔드 명세. 수치는 eval/zeroshot 오프라인 실측 (같은 val 300장, 같은 채점기).
BACKENDS = {
    "finetuned": {
        "label": "YOLOv8n 파인튜닝 (mAP50 0.999 · 7ms)",
        "short": "YOLOv8n fine-tuned",  # 검출 뷰 오버레이용 (cv2 는 한글 불가)
        "kind": "torchscript",
        "model": "models/best.torchscript",
        "meta": "models/model_meta.json",
        "conf": 0.5, "bridge_min_score": 0.5, "imgsz": 640,
    },
    "yoloworld": {
        "label": "YOLO-World v2 zero-shot (0.68 · 실시간)",
        "short": "YOLO-World v2 (zero-shot)",
        "kind": "torchscript",
        "model": "models/yoloworld_v2s.torchscript",
        "meta": "models/yoloworld_meta.json",
        "conf": 0.003, "bridge_min_score": 0.003, "imgsz": 1280,
        "hint": "파일이 없으면: training/.venv/bin/python training/export_yoloworld.py",
    },
    "gdino": {
        "label": "Grounding DINO zero-shot (0.88 · ~0.2s 비동기)",
        "short": "Grounding DINO (zero-shot)",
        "kind": "worker",
        "script": "isaac_ext/roboe_block_stacking/perception/gdino_worker.py",
        "python": "training/.venv/bin/python",
        "bridge_min_score": 0.25,
        "hint": "학습 venv 필요: training/.venv + eval/zeroshot/requirements-extra.txt",
    },
}

# GUI 드롭다운 항목 ("키 | 라벨" 형식 - 확장 쪽은 앞 토큰만 파싱한다)
DETECTOR_UI_ITEMS = [f"{key} | {cfg['label']}" for key, cfg in BACKENDS.items()]


def detector_key_from_ui(item: str) -> str:
    return str(item).split("|")[0].strip()


class _WorkerClient:
    """서브프로세스 검출 워커의 비동기 클라이언트 (단일 미결 요청 프로토콜)."""

    def __init__(self, python_path, script_path):
        self.log_path = Path(tempfile.gettempdir()) / "roboe_gdino_worker.log"
        self._log_f = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            [str(python_path), str(script_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log_f,
            text=True, bufsize=1,
        )
        self.ready = False
        self.dead = False
        self.load_s = None
        self._busy = False
        self._lock = threading.Lock()
        self.latest = ([], 0.0)          # (detections, ms)
        import atexit                    # Kit 종료 시 워커 고아 프로세스 방지
        atexit.register(self.stop)
        self._last_done_t = None
        self.effective_period_s = None   # 완성본 사이 간격 (실효 Hz 표기용)
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self):
        try:
            for line in self.proc.stdout:
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("ready"):
                    self.ready = True
                    self.load_s = msg.get("load_s")
                    carb.log_info(f"[ROBOE] gdino 워커 준비 ({msg})")
                    continue
                now = time.perf_counter()
                with self._lock:
                    dets = msg.get("detections", [])
                    for d in dets:
                        d["box"] = tuple(d["box"])
                    self.latest = (dets, float(msg.get("ms", 0.0)))
                    if self._last_done_t is not None:
                        self.effective_period_s = now - self._last_done_t
                    self._last_done_t = now
                    self._busy = False
                if msg.get("error"):
                    carb.log_warn(f"[ROBOE] gdino 워커 프레임 오류: {msg['error']}")
        finally:
            self.dead = True  # stdout EOF = 프로세스 종료

    def submit(self, rgb):
        """놀고 있을 때만 프레임을 넘긴다 (밀린 프레임은 버린다 - 최신이 정의상 더 좋다)."""
        if not self.ready or self.dead:
            return
        with self._lock:
            if self._busy:
                return
            self._busy = True
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            with self._lock:
                self._busy = False
            return
        try:
            self.proc.stdin.write(json.dumps(
                {"jpg_b64": base64.b64encode(jpg.tobytes()).decode()}) + "\n")
        except (BrokenPipeError, ValueError):
            self.dead = True

    def stop(self):
        try:
            if not self.dead:
                self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()
        finally:
            self._log_f.close()


class DetectorHub:
    def __init__(self, repo_root, report_fn=None):
        self.repo_root = Path(repo_root)
        self._report = report_fn or (lambda t: None)
        self.current_key = None
        self.current_label = "(미선택)"
        self.current_short = "(none)"
        self.bridge_min_score = 0.5
        self.active_detector = None   # torchscript 백엔드의 CubeDetector (worker 면 None)
        self.last_latency_ms = 0.0
        self.mode_note = ""           # 헤더에 붙는 상태 표기 ("비동기 2.1Hz" 등)
        self._worker = None

    # ------------------------------------------------------------------ 선택
    def select(self, key) -> str:
        if key not in BACKENDS:
            return f"알 수 없는 인식 소스: {key}"
        if key == self.current_key:
            return f"인식 소스 유지: {BACKENDS[key]['label']}"
        cfg = BACKENDS[key]

        if cfg["kind"] == "torchscript":
            model_path = self.repo_root / cfg["model"]
            if not model_path.exists():
                hint = cfg.get("hint", "")
                return f"모델 없음: {cfg['model']}\n-> {hint}" if hint else f"모델 없음: {cfg['model']}"
            meta_path = self.repo_root / cfg.get("meta", "")
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            try:
                det = CubeDetector(
                    str(model_path),
                    meta.get("class_names", _DEFAULT_CLASSES),
                    imgsz=int(meta.get("imgsz", cfg["imgsz"])),
                    conf=float(meta.get("conf", cfg["conf"])),
                )
            except Exception as exc:
                carb.log_warn(f"[ROBOE] 검출기 로드 실패({key}): {exc}")
                return f"검출기 로드 실패: {exc}"
            self._stop_worker()
            self.active_detector = det
            self.bridge_min_score = float(meta.get("bridge_min_score", cfg["bridge_min_score"]))
            self.mode_note = ""
        else:  # worker
            python_path = self.repo_root / cfg["python"]
            script_path = self.repo_root / cfg["script"]
            if not python_path.exists():
                return (f"학습 venv 없음: {cfg['python']}\n-> {cfg.get('hint', '')}")
            self._stop_worker()
            self.active_detector = None
            self._worker = _WorkerClient(python_path, script_path)
            self.bridge_min_score = float(cfg["bridge_min_score"])
            self.mode_note = " · 워커 로딩 중"

        self.current_key = key
        self.current_label = cfg["label"]
        self.current_short = cfg["short"]
        gate = self.bridge_min_score
        msg = (f"인식 소스 = {cfg['label']}\n"
               f"bridge 게이트 -> {gate:g} (모델의 신뢰도 보정 특성에 맞춤 - README §2.5)")
        if cfg["kind"] == "worker":
            msg += "\n워커(학습 venv) 기동 중... 준비되면 검출이 시작됩니다 (최초 ~10s)."
        carb.log_warn(f"[ROBOE] {msg.splitlines()[0]}")
        return msg

    # ------------------------------------------------------------------ 검출
    def detect(self, rgb):
        """현재 백엔드로 검출. worker 백엔드는 비동기 - '가장 최근 완성본'을 돌려준다."""
        if self.active_detector is not None:
            dets = self.active_detector(rgb)
            self.last_latency_ms = self.active_detector.last_latency_ms
            self.mode_note = ""
            return dets
        w = self._worker
        if w is None:
            return []
        if w.dead:
            self.mode_note = f" · 워커 종료됨 (로그: {w.log_path})"
            return []
        if not w.ready:
            self.mode_note = " · 워커 로딩 중"
            return []
        w.submit(rgb)
        dets, ms = w.latest
        self.last_latency_ms = ms
        hz = (1.0 / w.effective_period_s) if w.effective_period_s else None
        self.mode_note = f" · 비동기 {hz:.1f}Hz" if hz else " · 비동기"
        return dets

    # ------------------------------------------------------------------ 정리
    def _stop_worker(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

    def shutdown(self):
        self._stop_worker()
        self.active_detector = None
        self.current_key = None
