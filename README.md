# [Take-home task] AI Vision 기반 Block Stacking

Isaac Sim 5.1 시뮬레이터 안에서, 고정된 **ZED-X RGB-D 카메라**로 색상 큐브 4개를
**AI 검출 모델**로 인식하고, **Franka 로봇 암**이 **빨강 → 노랑 → 연두 → 파랑** 순서로
사용자가 지정한 위치에 쌓는 파이프라인입니다.

로봇은 시뮬레이터가 알고 있는 정답 좌표를 **읽지 않습니다.** 카메라와 AI가 만든
인식 결과만 보고 동작하므로, 인식 오차가 곧 파지 오차가 됩니다 — 쌓기가 성공한다는 것
자체가 인식 파이프라인이 동작한다는 증명입니다.

[![데모 영상 — 클릭하면 재생](media/demo/demo_thumbnail.png)](media/demo/demo_stacking.mp4)

> ▶ **데모 영상**: [`media/demo/demo_stacking.mp4`](media/demo/demo_stacking.mp4)
> (왼쪽: 3인칭 시점, 오른쪽: ZED-X 카메라 시점 + AI 검출 결과)

## 결과 요약

| 항목 | 결과 |
|---|---|
| 쌓기 성공률 — **기본 배치** (과제 명세 시나리오) | **5/5 연속 완주**, 회당 약 35초 |
| 쌓기 성공률 — **랜덤 배치** (위치+회전 무작위, 인식 초기값 평균 36cm 오염) | **10/10 (100%)**, 평균 44.9초 |
| 완성 탑의 중심 정렬 오차 | 최대 **7.4mm** (큐브 한 변 51.5mm) |
| AI 인식 → 3D 위치 오차 | 평균 **2.4mm** (95백분위 4.2mm) |
| 검출 정확도 (mAP50) / 추론 시간 | **0.995** / 프레임당 2.2ms |
| 인식 모델 교체 | GUI에서 4종 실행 중 전환 (파인튜닝 YOLO · YOLO-World · Grounding DINO · Qwen2.5-VL) |

---

## 1. 전체 구조 — 한 장으로 보기

파이프라인은 4단계입니다. 카메라 프레임이 들어올 때마다 ①→④를 약 10Hz로 반복합니다.

![전체 순서도](media/figures/pipeline_flowchart.png)

1. **① AI 인식** — RGB 이미지에서 색상 큐브를 찾아 클래스(색)와 바운딩박스를 출력한다.
2. **② 파지점 추정** — 바운딩박스와 깊이(Depth)로 큐브의 3D 위치와 회전(yaw)을 계산한다.
3. **③ 판단·제어** — 인식 결과를 필터링해 로봇의 세계 모델(belief)을 갱신하고,
   의사결정 로직이 다음 행동을 골라 로봇을 움직인다.
4. **④ 파지 판단** — 그리퍼 관절 폭으로 "잡았다/놓았다"를 스스로 판정한다.

> 순서도 원본: [`media/figures/pipeline_flowchart.mmd`](media/figures/pipeline_flowchart.mmd) (mermaid)

**베이스 코드**: Isaac Sim 내장 Franka **Cortex** 예제(Block Stacking, Apache-2.0).
Cortex 는 Isaac Sim 의 로봇 의사결정 프레임워크로, "로봇이 믿는 세계(belief)"와
"실제 물리 세계"를 구분해서 다룹니다. 이 과제의 통합 지점은 Cortex 가 인식 연동용으로
설계해 둔 `CortexObject.set_measured_pose()` API 하나이며, 의사결정 로직은 쌓기 순서
설정 외에 수정하지 않았습니다.

## 2. 단계별 상세

### 2.1 ① AI 인식 — 색상 큐브 검출

**모델: YOLOv8n 파인튜닝** (4클래스: red / yellow / green(연두) / blue)

| 후보 | 판단 |
|---|---|
| **YOLOv8n 파인튜닝** ✅ 채택 | 4클래스·고정 시점에 충분한 용량, 추론 수 ms, 합성 데이터로 라벨링 비용 0 |
| 색 임계처리 (HSV) | 과제의 "AI 모델 활용" 조건 미충족, 조명 변화에 취약 |
| Faster R-CNN | 추론 약 10배 느림 |
| Zero-shot 계열 | 학습이 필요 없다는 장점 → **§3 에서 4종을 실측 비교** (결론: 역할이 다름) |

실시간성이 중요한 이유: 인식이 시뮬레이션 루프 안에서 GPU 를 렌더링과 나눠 쓰기
때문에, 느린 모델은 제어 주기 자체를 망가뜨립니다 (단독 2.2ms → 렌더와 공유 시 27ms).

**학습 데이터: 합성 데이터 (Isaac Sim Replicator)** — 사람이 라벨링한 이미지 0장.
Replicator 는 시뮬레이터가 이미지를 렌더링하면서 정답 라벨(박스 좌표·클래스)까지 자동
생성해 주는 도구입니다. 런타임과 **같은 씬을 같은 카메라**로 찍어 학습하므로 학습-실전
차이(도메인 갭)가 원리적으로 없습니다. 큐브 배치·로봇 자세·조명·카메라 미세 지터를
무작위화해 train 2,800장 + val 300장(별도 시드)을 생성했습니다.

- 학습 결과: **mAP50 0.9949**, 노랑↔연두 오분류 600건 중 1건
- 배포: 학습은 별도 가상환경(ultralytics), 런타임은 **TorchScript** 로 내보내 Isaac Sim
  환경에 **추가 패키지 0개**로 로드. 내보낸 모델과 원본의 출력 일치를 검증함
  (33개 박스 전부 IoU 1.00000)

### 2.2 ② 파지점 추정 — 바운딩박스 → 3D 위치·자세

**위치** — 깊이 카메라는 큐브의 "카메라 쪽 표면(앞면)"까지의 거리를 줍니다.
그래서 3단계로 중심을 복원합니다:

1. 바운딩박스 **중앙 40% 영역의 깊이 중앙값** (가장자리·배경 픽셀 배제)
2. 박스 중심 픽셀을 **역투영** → 큐브 앞면의 3D 점
3. 시선 광선과 정육면체의 교점 공식 `t = h / max(|uₓ|,|u_y|,|u_z|)` 으로
   앞면→중심 보정 + 바닥 관통 방지 클램프

| | 위치 오차 (n=120) |
|---|---|
| 보정 없이 앞면 좌표 사용 | 평균 28.3mm — 파지 실패권 (반큐브 25.8mm 초과) |
| **보정 적용** | 평균 **2.4mm**, 95백분위 4.2mm |

**자세(yaw)** — 박스 안의 깊이 픽셀을 3D 로 역투영해 **큐브 상면만 골라 바닥에 투영**하고,
최소 외접 사각형(`cv2.minAreaRect`)의 기울기를 yaw 로 씁니다. 정육면체는 90° 대칭이므로
yaw 는 0~90° 범위로 정규화합니다. (큐브가 회전된 채 놓이면 정렬 가정 파지는 모서리를
치기 때문에 필요 — 랜덤 배치 시나리오에서 실측으로 확인된 요구사항)

### 2.3 ③ 판단·제어 — "로봇은 인식 결과만 본다" (belief 분리)

로봇에는 **렌더링을 끈 고스트 큐브**(belief)만 등록되어 있습니다. 인식 결과가 고스트를
움직이고, 로봇은 고스트를 보고 계획하며, 물리적으로는 진짜 큐브를 집습니다.
물리 큐브를 그대로 등록하면 인식을 꺼도 완벽히 동작해 버려서 — 인식이 실제로
기여하는지 증명할 수 없기 때문입니다.

인식 결과를 belief 에 반영하기 전에 **Cortex Bridge** 가 거릅니다.
핵심 원칙은 하나입니다 — **"조작 중에는 belief 를 건드리지 않는다."**

| 안전장치 | 막는 문제 |
|---|---|
| 신뢰도 게이트 + 클래스별 1개 선택 | 유령·중복 검출 |
| EMA 필터 + 데드밴드 8mm | 프레임 간 흔들림, 목표가 미세하게 계속 움직이는 문제 |
| 조작 중 동결 + 파지 시 강체 부착 | 잡으러 가는 중·운반 중에 목표가 튀는 문제 |
| 작업공간 게이트 | 가림·배경 깊이가 만드는 불가능한 3D 값 |
| 탑 보호 | 완성된 탑을 오검 하나가 무너뜨리는 문제 |

**행동 계층에서 해결한 것** (제어기 수정 없이 의사결정 계층에서):

- **운반 높이 상향(carry-high)**: 낮게 운반하면 장애물 회피력과 목표 인력이 상쇄돼
  엔드이펙터가 탑 앞에서 멈추는 정체가 생김 → 운반 경로를 높여 해소
- **최소 회전 파지 자세 선택**: 90° 대칭인 큐브의 등가 파지 자세 4개 중 손목 회전이
  가장 작은 것을 선택 → 손목이 관절 한계까지 감기는 교착을 제거

### 2.4 ④ 파지 판단 — 시뮬레이터에 묻지 않는다

"잡았는가"는 시뮬레이터 내부 정보가 아니라 **프로프리오셉션**(로봇 자신의 관절 상태)으로
판정합니다: 그리퍼 관절 폭이 큐브 크기(51.5mm)에서 안정되고 손끝이 목표 옆에 있으면
파지 성립 — 그 순간 belief 를 동결하고 큐브를 손에 강체 부착합니다. 그리퍼가 열리면
놓은 것으로 보고 부착을 해제합니다. 별도 캘리브레이션이 필요 없습니다.

## 3. 확장 — Zero-shot 인식 4종 비교 (학습 없이 "말로" 검출)

Zero-shot 검출기는 재학습 없이 **텍스트 프롬프트만으로** 새 물체를 찾는 모델입니다.
채택 모델을 포함한 4종을 **같은 검증 이미지 300장, 같은 채점 코드**로 비교했습니다.

| 모델 | 방식 | mAP50 | pick 정확도* | 지연/장 |
|---|---|---|---|---|
| **YOLOv8n 파인튜닝** ✅ 기본 | 폐쇄셋 학습 | **0.999** | **0.999** | **7 ms** |
| Grounding DINO (tiny) | zero-shot 검출기 | 0.878 | 0.873 | 177 ms |
| Qwen2.5-VL-3B | zero-shot 생성형 VLM | 0.820† | 0.867† | 3,852 ms |
| YOLO-World v2 (s) | zero-shot 실시간 검출기 | 0.680 | 0.697 | 13 ms |

(\*) pick 정확도 = "클래스별 최고 신뢰도 1개를 집는다"는 브리지 정책이 실제 그 색 큐브에
맞은 비율 — 쌓기 성공을 가장 직접 예측하는 지표. (†) 장당 약 4초라 30장 서브셋.
비교 그림: [`media/figures/zeroshot_compare.png`](media/figures/zeroshot_compare.png)

### 각 모델에 넣은 프롬프트

모델 계열마다 프롬프트의 형태 자체가 다릅니다 — 어휘 목록 → 캡션 → 지시문 순으로
언어 이해가 깊어지고, 그만큼 느려집니다.

**YOLO-World** — 클래스 어휘 목록. 실행 전에 한 번만 벡터로 변환해 모델에 구워 넣음(그래서 빠름):

```python
["red cube", "yellow cube", "light green cube", "blue cube"]
```

연두를 `"green cube"` 로 부르면 mAP50 0.525, `"light green cube"` 로 부르면 0.680 —
**프롬프트 단어 하나가 하이퍼파라미터**입니다.

**Grounding DINO** — 마침표로 구분한 캡션 한 줄. 매 프레임 텍스트를 함께 인코딩:

```python
"a red cube. a yellow cube. a green cube. a blue cube."
```

**Qwen2.5-VL** — 챗봇형 모델이라 출력 형식까지 지정한 지시문:

```python
"Detect every small colored cube in this image. There are up to four cubes: "
"red, yellow, green (light green), and blue. "
'Output ONLY a JSON array like [{"bbox_2d": [x1, y1, x2, y2], "label": "red cube"}] '
"with one entry per cube. No other text."
```

(파인튜닝 YOLOv8n 은 폐쇄셋이라 프롬프트가 없습니다 — 클래스가 학습으로 고정됨)

### 결론 — 대체가 아니라 역할 분담

- 이 과제처럼 **클래스가 고정되고 실시간 루프**가 필요한 곳 = 파인튜닝 소형 모델
  (zero-shot 은 연두↔노랑 같은 미세 색 구분에서 실제로 흔들리고, 신뢰도 값도 보정되어
  있지 않아 고정 게이트에 그대로 꽂을 수 없음)
- zero-shot 의 자리 = **신규 객체 프로토타이핑**, 그리고 정답 라벨이 없는 실환경에서
  **자동 라벨링(교사) → 소형 모델 학습(학생)**. 본 과제에서는 시뮬레이터가 교사였지만
  실환경에서는 zero-shot 이 그 자리를 채웁니다 — 두 접근은 한 파이프라인의 두 단계입니다.

**라이브 통합**: 4종 전부 GUI 드롭다운으로 실행 중 교체 가능하며(§6.1), 신뢰도 게이트는
백엔드에 맞춰 자동 전환됩니다. **Grounding DINO 만으로도 쌓기 완주**를 확인했습니다
(zero-shot 인식이 초당 2~3회로 느려도 belief 구조가 흡수).

## 4. 검증 결과

| 검증 | 결과 |
|---|---|
| 기하 체인 (AI 없이 정답 픽셀 왕복) | 오차 평균 0.8mm |
| TorchScript 출력 일치 | IoU 1.00000 (33/33 박스), 클래스 불일치 0 |
| 인식 정확도 (정지 큐브 120회 측정) | 평균 2.4mm / 95백분위 4.2mm / 미검출 0 |
| **쌓기 — 기본 배치** (과제 명세) | **5회 연속 완주**, 약 35초, 탑 오차 1~8mm |
| 쌓기 — 대체 적재 위치 | 완주 / 제약 위반 위치는 사전 거부 |
| **쌓기 — 랜덤 스트레스** (10회) | **10/10 (100%)**, 평균 44.9초, 탑 오차 최대 7.4mm |

랜덤 스트레스는 과제 범위를 넘어선 조건입니다: 매회 큐브를 무작위 위치·회전으로 흩고
(작업공간 반경 0.40~0.75m, yaw 0~90°), **belief 를 평균 36cm 틀린 상태로 시작**시켜
인식이 스스로 교정해야만 성공하게 했습니다 — "인식이 진짜 루프 안에 있다"의 정량 증명.

그림: [`media/figures/trials_summary.png`](media/figures/trials_summary.png) ·
[`media/figures/ablation_depth_correction.png`](media/figures/ablation_depth_correction.png) ·
원자료: `media/m6/trials.csv`

## 5. 설치

**요구 환경**: Ubuntu 22.04+, NVIDIA RTX GPU(개발·검증: RTX 4080 SUPER 16GB),
드라이버 550+, 디스크 약 60GB, 최초 실행 시 인터넷(에셋 다운로드).
모든 스크립트는 저장소 상대 경로만 사용합니다 — 어디에 체크아웃해도 동작합니다.

### 5.1 Isaac Sim (최초 1회, 수동)

```bash
conda create -n isaacsim python=3.11 -y
conda activate isaacsim
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
git clone <저장소> roboe_block_stacking && cd roboe_block_stacking
```

### 5.2 런타임 설치 (스크립트)

```bash
conda activate isaacsim
bash scripts/setup_runtime.sh   # 추론 의존성 + GUI 예제 등록 + 자가 검증. 재실행 안전
```

학습된 모델(`models/best.torchscript`)이 저장소에 포함되어 있어 **여기까지만 하면 바로
실행 가능**합니다. Isaac Sim 최초 실행 시 EULA 동의와 에셋 다운로드(수 분)가 있습니다.

### 5.3 zero-shot 백엔드 (선택)

기본 백엔드만 쓸 거라면 생략해도 됩니다.

```bash
bash scripts/setup_zeroshot.sh  # 학습 venv + YOLO-World 재생성(657MB). 재실행 안전
```

Grounding DINO(~700MB)·Qwen2.5-VL(~7GB) 가중치는 첫 사용 시 자동 다운로드됩니다.

### 5.4 학습 재현 (선택)

```bash
python sdg/generate_dataset.py --train 2500 --val 300              # isaacsim 환경
python sdg/generate_dataset.py --train 300 --val 0 --start-index 2500 --held-prob 0.9 --seed 777
training/.venv/bin/python training/train.py                         # 학습 + export
training/.venv/bin/python training/verify_torchscript_decode.py     # 배포 전 검증
```

zero-shot 오프라인 비교 재현: `eval/zeroshot/run_*.py` → `summarize.py` (전부 학습 venv).

## 6. 실행

### 6.1 GUI (권장)

```bash
conda activate isaacsim && isaacsim
```

**Window → Examples → Robotics Examples → Custom → ROBOE Block Stacking**
→ `Tower X/Y` 입력 → `LOAD` → `Start`

컨트롤 패널 기능:

- **Perception On/Off** — 끄면 로봇이 기본 위치의 고스트로 갑니다 (인식 의존성을 눈으로 확인)
- **Randomize Cubes** — 실행 중 안전한 위치로 큐브 재배치. 인식이 재검출로 따라잡는
  과정이 그대로 보입니다
- **AI 검출 뷰 창** — 검출기가 보는 이미지 + 박스/점수 실시간 표시
- **인식 소스 드롭다운** — 파인튜닝 YOLO / YOLO-World / Grounding DINO / Qwen2.5-VL 을
  실행 중 전환. 게이트 자동 조정, 전환 시 이전 백엔드를 먼저 내려 VRAM 순차 점유
- **Perception 패널** — 검출·3D 좌표·yaw·지연·브리지 판정 실시간

### 6.2 GUI 없이 (검증·평가)

```bash
python standalone/run_stacking_perception.py            # 인식 기반 E2E 쌓기
python standalone/run_stacking_perception.py --tower 0.45 0.25
python eval/run_trials.py --trials 10 --record          # 배치 평가 + 영상
python standalone/record_demo.py                        # 데모 영상 재생성
python standalone/verify_m1_camera.py                   # 기하 체인 검증
python standalone/verify_m4_perception.py               # 인식 정확도 측정
python eval/test_decode_math.py                         # 단위 테스트 (GPU 불필요)
```

## 7. 저장소 구조

| 경로 | 내용 |
|---|---|
| `isaac_ext/roboe_block_stacking/` | Isaac Sim GUI 예제 (씬 + 행동 + 인식) |
| ├ `behavior/` | Block Stacking 의사결정 로직 (스톡 복사본, 변경점 `[ROBOE]` 주석) |
| ├ `perception/` | 카메라 · 검출기 4종 · 3D 추정 · Cortex Bridge |
| └ `scene_setup.py` | 큐브/조명/고스트/적재 위치 검증 (학습 데이터 생성과 공용) |
| `scripts/` | 설치 스크립트 (`setup_runtime.sh`, `setup_zeroshot.sh`) |
| `sdg/` | Replicator 합성 데이터셋 생성 |
| `training/` | YOLO 학습·검증·export (별도 venv) |
| `models/` | 학습된 모델 (TorchScript 포함 — 바로 실행 가능) |
| `standalone/` | E2E 러너·검증 스크립트 |
| `eval/` | 배치 평가, zero-shot 비교, 단위 테스트, 그림 생성 |
| `media/` | 데모 영상 · 그림 · 원자료 CSV |

## 8. 명세 해석 (가정 명시)

- **"집기 순서" = 쌓는 순서**로 해석: 먼저 집는 빨강이 탑의 맨 아래.
  (반대 해석이면 `behavior/block_stacking_behavior.py` 의 `order_preference` 한 줄로 대응)
- **"연두"**: 베이스 예제의 `GreenCube` 이름은 유지하고 **색만** 연두로 조정
  (노랑과 가장 잘 구분되는 값을 렌더 픽셀로 선정, 혼동행렬로 검증)
- **적재 위치**: GUI 입력 + 사전 유효성 검증 (로봇 도달 범위 0.30~0.75m,
  큐브 스폰 영역과 0.15m 초과 이격)

## 9. 한계와 향후 과제

- **깊이는 렌더 깊이**: 시뮬레이터의 정확한 깊이를 사용 — 실물 스테레오 깊이의
  노이즈·홀은 미반영. 실물 전환 시 `zed_camera.py` 의 데이터 소스만 ZED SDK 스트림으로
  교체하면 되는 구조이나(계층 분리), 깊이 필터링 추가가 필요할 것
- **고정 카메라 1대**: 가림이 길어지면 해당 큐브의 belief 는 마지막 관측에 머무름
- **클래스별 1개 가정**: 같은 색 큐브가 여러 개인 씬은 지원하지 않음
- **yaw 만 추정**: 큐브가 바닥에 똑바로 놓였다고 가정 (기울어진 큐브는 범위 밖)
- **zero-shot 백엔드는 시연·분석용 프로필**: YOLO-World 는 신뢰도 보정 문제로 라이브
  검출이 불안정하고, Qwen 은 초당 0.2회 수준 — 기본 백엔드의 대체가 아님 (§3 결론)

## 10. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| 첫 LOAD 가 수 분 걸림 | 클라우드 에셋(ZED-X 등) 최초 다운로드 — 1회만 발생 |
| 인식 소스에서 YOLO-World 선택 시 에러 | 657MB 모델이 git 미포함 — `bash scripts/setup_zeroshot.sh` 로 재생성 |
| GDINO/Qwen 선택 후 한동안 검출 없음 | 첫 선택 시 가중치 다운로드+로딩 (수십 초~수 분). 로그: `/tmp/roboe_gdino_worker.log`, `/tmp/roboe_qwen_worker.log` |
| Qwen 선택 시 메모리 부족 | VRAM 약 7GB 필요 — 다른 GPU 점유 프로세스 종료 후 재시도 |
| `eval/benchmark_detector.py` 실패 | 검증 이미지 필요 — §5.4 의 데이터 생성 먼저 실행 |

---

베이스: Isaac Sim Franka Cortex Examples (Apache-2.0). 과제 명세 PDF 는 저장소에
포함하지 않습니다(발제사 소유).
