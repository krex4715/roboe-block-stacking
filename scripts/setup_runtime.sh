#!/usr/bin/env bash
# [ROBOE] 런타임 설치 — isaacsim conda 환경을 활성화한 상태에서 실행한다.
#   conda activate isaacsim_roboe && bash scripts/setup_runtime.sh
#
# 하는 일: ① 추론 의존성 설치 ② GUI 예제 등록(심링크) ③ 단위 테스트로 자가 검증.
# 재실행해도 안전하다(idempotent). Isaac Sim 자체 설치는 README §5.1 참조.
set -euo pipefail
cd "$(dirname "$0")/.."

# 0) NVIDIA Omniverse EULA 동의 등록 + 환경 확인.
#    최초 `import isaacsim` 시 EULA 대화형 프롬프트가 뜨는데, 스크립트/헤드리스에서는
#    입력을 받을 수 없어 실패한다 (클린 환경에서 실측). 이 스크립트 실행을 동의로
#    처리하고, conda 활성화 훅에 등록해 이후 GUI/헤드리스 실행에도 적용한다.
#    EULA 전문: https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo 'export OMNI_KIT_ACCEPT_EULA=YES' > "$CONDA_PREFIX/etc/conda/activate.d/roboe_eula.sh"
echo "[i] NVIDIA Omniverse EULA 동의 처리 + conda 활성화 훅 등록 (OMNI_KIT_ACCEPT_EULA=YES)"

if ! python -c "import isaacsim" 2>/dev/null; then
    echo "[ERR] 'import isaacsim' 실패 — isaacsim_roboe 환경인지 확인하세요 (conda activate isaacsim_roboe)." >&2
    echo "      Isaac Sim 미설치라면 README '직접 실행하기' 1) 을 먼저 실행하세요." >&2
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
