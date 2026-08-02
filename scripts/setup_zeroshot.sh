#!/usr/bin/env bash
# [ROBOE] zero-shot 인식 백엔드 설치 — YOLO-World / Grounding DINO / Qwen2.5-VL.
#   bash scripts/setup_zeroshot.sh
#
# 기본 백엔드(파인튜닝 YOLOv8n)만 쓸 거라면 이 스크립트는 필요 없다.
# isaacsim 환경과 무관하게 동작한다(전부 training/.venv 에 설치 — 신규 패키지 0 원칙).
# 재실행해도 안전하다(idempotent).
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) 학습 venv (없으면 생성) + 의존성
[ -d training/.venv ] || python3 -m venv training/.venv
training/.venv/bin/pip install -r training/requirements.txt
training/.venv/bin/pip install -r eval/zeroshot/requirements-extra.txt

# 2) CLIP — YOLO-World 텍스트 인코더.
#    ultralytics 의 자동 설치는 venv 밖(sys 파이썬)으로 새는 경우가 있어 명시 설치한다.
training/.venv/bin/python -c "import clip" 2>/dev/null || \
  training/.venv/bin/pip install git+https://github.com/ultralytics/CLIP.git

# 3) YOLO-World TorchScript 재생성 (657MB 라 git 미포함. 프롬프트 임베딩을 bake 하고
#    ultralytics 와의 parity 검증까지 수행 — 실패 시 여기서 멈춘다)
[ -f models/yoloworld_v2s.torchscript ] || \
  training/.venv/bin/python training/export_yoloworld.py

echo "[OK] zero-shot 백엔드 설치 완료."
echo "     Grounding DINO(~700MB) / Qwen2.5-VL(~7GB) 가중치는 첫 사용 시 자동 다운로드된다."
echo "     GUI 컨트롤 패널의 '인식 소스' 드롭다운에서 전환 (README §4.1)."
