# [Take-home Task] AI Vision 기반 Block Stacking

Isaac Sim 5.1 위에서, 고정된 **ZED-X RGB-D 카메라**와 **AI 검출 모델**만으로 색상 큐브
4개를 인식해, **Franka 암**이 **빨강 → 노랑 → 연두 → 파랑** 순서로 사용자가 지정한
위치에 쌓는 프레임워크. 전제 — 로봇은 시뮬레이터가 알고 있는 정답 좌표를 읽지 않는다

**제출물 3종 바로가기** :
[**소스코드**](소스코드.md) ·
[**구현 설명 문서**](구현설명문서.md) ·
[**실행 영상**](실행영상.md)



## 결과 — 인식 소스 4종 × 배치 2종, 각 10회 실측

| 기본 배치 (과제 명세 그대로) | 랜덤 배치 (스트레스) |
| :---: | :---: |
| [![기본 배치 데모](media/demo/demo_default.gif)](media/demo/demo_default.mp4) | [![랜덤 배치 데모](media/demo/demo_random.gif)](media/demo/demo_random.mp4) |
| 큐브 4개 일자 정위치 | 위치·회전 무작위 + 초기 belief 를 평균 36cm 오염시켜 시작 (인식이 교정해야 성공하는 조건) |

(움직이는 미리보기 = GIF. 클릭하면 고화질 mp4 원본)

같은 파이프라인에서 인식 소스만 바꿔 두 배치를 각 10회씩 돌린 성공률.
랜덤 배치는 seed 를 고정해서 4개 소스가 동일한 10개 배치로 평가됨:

| 인식 소스 | 기본 배치 | 랜덤 배치 |
| --- | :---: | :---: |
| **YOLOv8n 파인튜닝** ✅ 기본 | **10/10** | **10/10** |
| Grounding DINO (tiny) | **10/10** | 4/10 |
| Qwen2.5-VL-3B | **10/10** | **10/10** |
| YOLO-World v2 (s) | 6/10 | 7/10 |

인식 소스별 대표 영상. 4편 모두 같은 랜덤 배치에서 실제 완주한 회차라 소스 간
직접 비교가 됨 (ZED 카메라 시점 + 검출 오버레이):

| YOLOv8n 파인튜닝 (10/10 · 10/10) | Grounding DINO (10/10 · 4/10) |
| :---: | :---: |
| [![YOLOv8n 파인튜닝](media/demo/demo_random.gif)](media/demo/demo_random.mp4) | [![Grounding DINO](media/demo/infer_gdino.gif)](media/demo/infer_gdino.mp4) |
| **Qwen2.5-VL-3B (10/10 · 10/10)** | **YOLO-World v2 (6/10 · 7/10)** |
| [![Qwen2.5-VL](media/demo/infer_qwen.gif)](media/demo/infer_qwen.mp4) | [![YOLO-World](media/demo/infer_yoloworld.gif)](media/demo/infer_yoloworld.mp4) |

- 실패 13회는 전부 특정 큐브 재검출 실패로 인한 정체 후 타임아웃(100초). 완성된 탑이
  무너진 사례는 없음 (오검출은 브리지 안전장치가 차단, Qwen 은 회당 최대 45회 기각)
- 파인튜닝 기준 완주 시간: 기본 평균 25.6초 / 랜덤 평균 22.0초. 완성 탑의 중심 정렬
  오차 최대 5.3mm (큐브 한 변 51.5mm), AI 인식 → 3D 위치 오차 평균 2.4mm
- 원자료: `media/e2e/<소스>_<배치>/trials.csv` (회당 시간·파지 횟수·게이트 통계)

## 전체 흐름

카메라 프레임이 들어올 때마다 ①→④를 반복함. 단계별 로직 상세는 각 링크 문서에.

- **[① AI 인식](docs/01_perception.md)** — RGB 이미지에서 큐브의 클래스(색)와 바운딩박스를 검출
- **[② 파지점 추정](docs/02_grasp_point.md)** — 바운딩박스 + 깊이(Depth)로 3D 위치(x, y, z)와 자세(yaw)를 복원.
  깊이는 큐브 "앞면"까지의 거리이므로 광선-정육면체 교점 공식으로 중심을 보정
  (보정 전 오차 28.3mm → 보정 후 2.4mm)
- **[③ 판단·제어](docs/03_decision_control.md)** — 인식 결과를 안전장치(신뢰도 게이트 · EMA 필터 · 조작 중 동결 ·
  작업공간 게이트 · 탑 보호)로 걸러 로봇의 믿음(belief)을 갱신하고, 의사결정 로직이
  다음 행동을 골라 로봇을 움직임
- **[④ 파지 판단](docs/04_grasp_state.md)** — 시뮬레이터에 묻지 않고, 그리퍼 관절 폭(프로프리오셉션)으로
  "잡았다/놓았다"를 스스로 판정

![전체 순서도](media/figures/pipeline_flowchart.png)

베이스는 Isaac Sim 내장 Franka **Cortex** 예제(Block Stacking, Apache-2.0).
통합 지점은 Cortex 가 인식 연동용으로 제공하는 `CortexObject.set_measured_pose()`
API 하나이고, 의사결정 로직은 쌓기 순서 설정 외에는 수정하지 않음.

## 핵심 — AI 인식 4종 실측 비교

과제의 핵심인 ① AI 인식을 **네 가지 방식으로 구현**하고, 같은 검증 이미지 300장 ·
같은 채점 코드로 실측 비교함. 네 방식 모두 GUI 드롭다운으로 **실행 중 전환**됨.

| 인식 소스 | 방식 | mAP50 | pick 정확도* | 지연/장 |
| --- | --- | --- | --- | --- |
| **YOLOv8n 파인튜닝** ✅ 기본 | 폐쇄셋 학습 (합성 데이터) | **0.999** | **0.999** | **7 ms** |
| Grounding DINO (tiny) | zero-shot 검출기 | 0.878 | 0.873 | 177 ms |
| Qwen2.5-VL-3B | zero-shot 생성형 VLM | 0.820† | 0.867† | 3,852 ms |
| YOLO-World v2 (s) | zero-shot 실시간 검출기 | 0.680 | 0.697 | 13 ms |

(\*) pick 정확도 = "클래스별 최고 신뢰도 1개를 집는다"는 정책이 실제 그 색 큐브에 맞은
비율. 쌓기 성공을 가장 직접 예측하는 지표. (†) 장당 약 4초라 30장 서브셋으로 측정.
비교 그림: [`media/figures/zeroshot_compare.png`](media/figures/zeroshot_compare.png)

분석 요약:

- 파인튜닝을 기본으로 채택. 클래스가 고정된 실시간 제어 루프 조건에서 전 지표 우위.
  학습 데이터는 Replicator 합성 train 2,800장 (수동 라벨링 0장, mAP50 0.9949),
  배포는 TorchScript 라 Isaac Sim 환경에 추가 패키지 없음
- zero-shot 공통 약점: 연두↔노랑 같은 미세 색 구분에서 흔들림 (연두→노랑 오인 67회),
  프롬프트 민감도가 큼 ("green"→"light green" 변경만으로 mAP50 0.525→0.680),
  신뢰도 보정이 안 되어 있어 고정 threshold 를 그대로 못 씀
- 오프라인 지표와 E2E 성공률이 일치하지 않음 (위 결과 표). 기본 배치는 mAP 순서와
  비슷하지만 (GDINO/Qwen 10/10, YOLO-World 6/10), 랜덤 배치에서는 mAP 0.878 인
  GDINO 가 4/10 로 떨어지고 가장 느린 Qwen(~0.2Hz)이 10/10. 정확도/지연/신뢰도
  보정/안전장치가 함께 작용하기 때문에 E2E 실측 없이는 판단이 어려움. YOLO-World
  실패는 대부분 3번째 큐브(연두) 재검출 정체로, 오프라인에서 확인된 연두 혼동과
  원인이 같음
- 결론은 대체가 아니라 역할 분담. 정답 라벨이 없는 실환경에서는 zero-shot 을
  auto-labeling 에 쓰고, 소형 파인튜닝 모델을 실시간 추론에 쓰는 구성이 자연스러움

프롬프트 원문과 채점 코드는 `eval/zeroshot/`, 인식 파이프라인 구현은
`isaac_ext/roboe_block_stacking/perception/` 에 있음.

## 직접 실행하기 (inference 재현)

**요구 환경**: Ubuntu 22.04+, NVIDIA RTX GPU(검증: RTX 4080 SUPER 16GB), 드라이버 550+,
최초 실행 시 인터넷. 모든 스크립트는 저장소 상대 경로만 사용함.

```bash
# 1) Isaac Sim (최초 1회)
conda create -n isaacsim_roboe python=3.11 -y && conda activate isaacsim_roboe
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# 2) 런타임 설치 (NVIDIA Omniverse EULA 동의 포함) — 학습된 모델이 저장소에
#    포함되어 있어 여기까지면 바로 실행 가능
bash scripts/setup_runtime.sh

# 3) zero-shot 백엔드 3종
bash scripts/setup_zeroshot.sh
```

**GUI 실행**: `isaacsim` → Window → Examples → Robotics Examples → Custom →
**ROBOE Block Stacking** → `Tower X/Y` 입력 → `LOAD` → `Start`

![GUI 안내 — 예제 위치 · 컨트롤 패널 · AI 검출 뷰](media/figures/gui_guide_full.png)

컨트롤 패널 확대 (AI Source / LOAD / RANDOMIZE / START):

![컨트롤 패널 확대](media/figures/gui_guide_panel.png)

- **인식 소스 드롭다운** — 4종을 실행 중 전환 (신뢰도 게이트 자동 조정, GDINO/Qwen 은
  첫 선택 시 가중치 자동 다운로드)
- **AI 검출 뷰 창** — 검출기가 보는 이미지 + 박스/점수 실시간 표시
- **Randomize Cubes** — 실행 중 큐브 재배치 → 인식이 재검출로 따라잡는 과정 확인
- **Perception On/Off** — 끄면 로봇이 기본 위치의 고스트로 감 (인식 의존성 확인)

**GUI 없이 (배치 평가·재현)**:

```bash
python standalone/run_stacking_perception.py                  # 기본 배치 E2E 쌓기
python eval/run_trials.py --backend gdino --layout random     # 소스/배치 골라 10회 평가
bash eval/run_e2e_matrix.sh                                   # 위 성공률 매트릭스 전체 재현
```

zero-shot 성능 비교 재현: `eval/zeroshot/run_*.py` → `summarize.py` (학습 venv,
`scripts/setup_zeroshot.sh` 가 구성).

---

베이스: Isaac Sim Franka Cortex Examples (Apache-2.0). 과제 명세 PDF 는 저장소에
포함하지 않음 (발제사 소유).
