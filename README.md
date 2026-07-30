# [Take-home task] AI Vision 기반 Block Stacking

Isaac Sim 5.1 환경에서 ZED-X RGB-D 카메라와 AI 검출 모델(YOLO)로 색상 큐브를 인식하고,
Franka 로봇 암이 **빨강 → 노랑 → 연두 → 파랑** 순서로 사용자 지정 위치에 쌓는 파이프라인.

> 베이스: Isaac Sim 내장 **Franka Cortex Examples** (Block Stacking behavior, Apache-2.0)

## 전체 구조 (Vision → Control)

```
ZED-X 고정 카메라 ──RGB──▶ YOLOv8n(TorchScript, in-process) ──bbox+cls──▶
        └────Depth───────────────────────────────▶ 3D 추정기 ──(p,q)──▶
Cortex Bridge(게이팅·필터) ──set_measured_pose()──▶ CortexObject(belief 동기화)
──▶ Block Stacking Decider Network(순서 수정) ──▶ MotionCommander(RMPFlow) ──▶ Franka
```

*(작성 중 — 제출 시 상세 설명, 모델 선택 이유, 실행 방법, 결과 수치가 채워집니다)*

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `isaac_ext/roboe_block_stacking/` | Isaac Sim GUI 예제 (씬 + behavior + perception) |
| `sdg/` | Replicator 합성 데이터셋 생성 |
| `training/` | YOLO 학습·검증·export (별도 venv) |
| `models/` | 학습된 가중치 (pt / torchscript / onnx) |
| `standalone/` | GUI 없이 실행하는 러너 (배치 평가용) |
| `eval/` | 정량 평가 스크립트 (3D 오차, 성공률) |
| `media/` | 실행 영상·스크린샷 |
