# [ROBOE] 비전 파이프라인 패키지
#
# 이 패키지는 두 가지 방식으로 import될 수 있도록 내부에서 상대 import만 사용한다:
#   1) Isaac Sim GUI 확장 경로: isaacsim.examples.interactive.user_examples.roboe_block_stacking.perception
#   2) standalone 스크립트: sys.path에 roboe_block_stacking을 추가한 뒤 `import perception`
#
# 모듈 구성:
#   zed_camera.py    ZED-X 스폰 · 배치 · RGB/Depth 취득
#   estimator_3d.py  픽셀+깊이 -> 3D 월드좌표 (큐브 중심 보정 포함)
#   detector.py      (M4) YOLO TorchScript 추론
#   cortex_bridge.py (M5) 인식 결과를 CortexObject에 발행
