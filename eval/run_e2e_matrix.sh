#!/usr/bin/env bash
# [ROBOE] E2E 성공률 매트릭스 - 인식 소스 4종 x 배치 2종 x 10회.
#   conda activate isaacsim && bash eval/run_e2e_matrix.sh
#
# 조합마다 SimulationApp 프로세스를 새로 띄운다 (GPU 상태 격리 - 특히 VLM 워커의
# VRAM 이 다음 조합으로 새지 않게). 같은 --seed 라 random 배치의 스폰 시퀀스가
# 백엔드 간 동일하다 = 통제 변인. 한 조합이 실패해도 다음 조합은 계속 진행한다.
# 결과: media/e2e/<backend>_<layout>/trials.csv + trial0 ZED 영상(원본/오버레이) + .log
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p media/e2e

for backend in finetuned yoloworld gdino qwen; do
  for layout in default random; do
    echo "=== $backend / $layout ($(date +%H:%M:%S)) ==="
    python -u eval/run_trials.py --backend "$backend" --layout "$layout" \
      --trials "${TRIALS:-10}" --record \
      >"media/e2e/${backend}_${layout}.log" 2>&1
    tail -n 4 "media/e2e/${backend}_${layout}.log"
  done
done

echo "=== 매트릭스 완료 ($(date +%H:%M:%S)) ==="
grep -h "BATCH_RESULT" media/e2e/*.log
