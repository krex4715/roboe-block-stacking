#!/usr/bin/env bash
# [ROBOE] 런타임 설치 — isaacsim conda 환경을 활성화한 상태에서 실행한다.
#   conda activate isaacsim && bash scripts/setup_runtime.sh
#
# 하는 일: ① 추론 의존성 설치 ② GUI 예제 등록(심링크) ③ 단위 테스트로 자가 검증.
# 재실행해도 안전하다(idempotent). Isaac Sim 자체 설치는 README §5.1 참조.
set -euo pipefail
cd "$(dirname "$0")/.."

# 0) 환경 확인 — isaacsim 환경이 아니면 중단
if ! python -c "import isaacsim" 2>/dev/null; then
    echo "[ERR] isaacsim 환경이 아닙니다. 'conda activate isaacsim' 후 다시 실행하세요." >&2
    exit 1
fi

# 1) 추론 런타임 의존성 (학습용 ultralytics 는 여기 설치하지 않는다 — README §2.1)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python-headless==4.11.0.86

# 2) GUI 예제 등록 (심링크 + import 한 줄 — 재실행해도 중복되지 않음)
UE="$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/exts/isaacsim.examples.interactive/isaacsim/examples/interactive/user_examples"
ln -sfn "$(pwd)/isaac_ext/roboe_block_stacking" "$UE/roboe_block_stacking"
grep -q RoboeBlockStackingExtension "$UE/__init__.py" 2>/dev/null || \
  echo "from isaacsim.examples.interactive.user_examples.roboe_block_stacking import RoboeBlockStackingExtension" >> "$UE/__init__.py"

# 3) 자가 검증 (GPU/Isaac Sim 불필요)
python eval/test_decode_math.py

echo "[OK] 런타임 설치 완료 — 'isaacsim' 실행 후 Window > Examples > Robotics Examples > Custom > ROBOE Block Stacking"
