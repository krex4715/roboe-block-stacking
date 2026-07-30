# [ROBOE Take-home] AI Vision 기반 Block Stacking — 커스텀 예제 패키지
#
# Isaac Sim 내장 "Franka Cortex Examples"(Apache-2.0, NVIDIA)를 베이스로 한 수정본.
# user_examples 슬롯에 심링크되어 GUI 예제 브라우저의 Custom 카테고리에 등록된다.
#
# 구성:
#   roboe_stacking_extension.py  브라우저 등록 + UI (Load/Start/Reset, 진단 패널)
#   roboe_stacking_example.py    씬 구성(Franka, 큐브 4개[, ZED-X]) + 실행 로직
#   behavior/                    Block Stacking decider network (스톡 복사본 → 순서/타워 수정 예정)
#   perception/                  비전 파이프라인 (카메라, YOLO 검출, 3D 추정, Cortex bridge)

from isaacsim.examples.interactive.user_examples.roboe_block_stacking.roboe_stacking_example import RoboeBlockStacking
from isaacsim.examples.interactive.user_examples.roboe_block_stacking.roboe_stacking_extension import (
    RoboeBlockStackingExtension,
)
