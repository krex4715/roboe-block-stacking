# [Take-home task] AI Vision 기반 Block Stacking

Isaac Sim 5.1 환경에서 **ZED-X RGB-D 카메라**와 **AI 검출 모델(YOLOv8n)** 로 색상 큐브를 인식하고,
Franka 로봇 암이 **빨강 → 노랑 → 연두 → 파랑** 순서로 사용자 지정 위치에 쌓는 파이프라인.

> 베이스: Isaac Sim 내장 **Franka Cortex Examples** (Block Stacking behavior, Apache-2.0)

**핵심 결과** — 인식 기반 end-to-end 스태킹 완주 (~35초, 탑 중심 오차 1~8mm).
로봇은 시뮬레이터의 ground truth 를 읽지 않고 **카메라가 만든 belief 만 보고** 동작한다.

---

## 1. 전체 구조 (Vision → Perception → Manipulation)

```
[Isaac Sim 씬]
   ZED-X (고정, Translate 1.0/0.0/1.0, Orient 0.0/-50.0/180.0, Kinematic)
        │
        ├── RGB  ──────▶ YOLOv8n (TorchScript, in-process, ~10Hz) ──▶ bbox + 클래스
        │                                                                │
        └── Depth ────────────────────────────────────────────────────┤
                        (distance_to_image_plane)                        ▼
                                                          3D 추정기 (estimator_3d)
                                                 bbox 중앙 깊이 중앙값 → 역투영 → 중심 보정
                                                                         │ (p, q)
                                                                         ▼
                                                     Cortex Bridge (게이팅·부착·보호)
                                                                         │
                                        ┌────────────────────────────────┘
                                        ▼
   [Belief 세계]  고스트 큐브 4개 (렌더링 OFF, 물리 없음)  ← 로봇이 보는 유일한 세계
                                        │
                                        ▼
                          Block Stacking Decider Network (Cortex, 순서만 수정)
                                        │
                                        ▼
                          MotionCommander (RMPFlow) → Franka → [물리 세계의 진짜 큐브를 집음]
```

### 왜 belief 를 분리했나 (고스트 구조)

물리 큐브를 그대로 behavior 에 등록하면 belief == 실제가 되어 **인식을 꺼도 완벽히
동작한다** — 인식이 기여하는지 증명할 수 없다. 또한 실측 결과, 그 구조에서는 인식
동기화가 물리 큐브를 순간이동시켜 조작 자체를 교란한다(발행 32회만으로 실패).

그래서 로봇에는 **렌더링을 끈 고스트 큐브**만 등록한다. 로봇은 고스트(=인식 결과)만
보고 계획하고, 물리적으로는 진짜 큐브를 집는다. **인식 오차가 곧 파지 오차**가 되므로
파이프라인이 실제로 동작함이 구조적으로 증명된다. 배치 평가에서는 매 트라이얼
belief 를 의도적으로 틀리게 시작시켜(평균 수십 cm) 인식이 스스로 교정해야만 성공하게 했다.

연결 지점은 Cortex 가 인식 연동용으로 설계해 둔 `CortexObject.set_measured_pose()` —
기본 설치본에는 호출처가 한 곳도 없는 API 다. 의사결정 로직은 쌓기 순서
(`order_preference`) 한 줄 외에 수정하지 않았다.

### 설계 원칙 — "조작 중에는 belief 를 건드리지 않는다"

인식은 로봇이 물체를 만지고 있지 않을 때만 권위를 갖는다. 12회의 통합 실험에서
실패 원인을 계측으로 규명하며 도달한 원칙이다 (상세: 커밋 히스토리와 코드 주석):

| Cortex Bridge 안전장치 | 막는 실패 모드 (전부 실측) |
|---|---|
| 신뢰도 게이트 + 클래스별 1개 | 유령/중복 검출 |
| EMA 필터 + 놓을 때 리셋 | 프레임 간 지터, 파지 전 잔존값 |
| **데드밴드 8mm** | 목표가 미세하게 계속 움직여 도달 판정이 영원히 안 남 |
| **조작 중 동결 + 프로프리오셉션 강체 부착** | 접근 중 목표 흔들림 / 운반 중 belief 정지 |
| **작업공간 게이트** | 가림 중 오검 + 배경 깊이가 만드는 불가능한 3D 추정 |
| **탑 보호 (무예외)** | 배치 직후 오검 하나가 완성 탑의 belief 를 파괴 |
| z 클램프 | 지면 관통 |

파지 감지는 시뮬레이터 정보가 아니라 **프로프리오셉션**으로 한다 — 그리퍼 관절 폭이
큐브 크기(51.5mm)에서 안정되고 EE 가 고스트 옆에 있으면 물리적 파지 성립. 그 순간의
고스트-EE 상대 자세를 측정·기록해 강체 부착하므로 TCP 캘리브레이션이 필요 없다.

## 2. 인식 방식 / 모델 선택 이유

### 2.1 왜 학습 기반 검출기(YOLOv8n)인가

| 후보 | 판단 |
|---|---|
| **YOLOv8n 파인튜닝** ✅ 채택 | 4클래스·고정 시점에 충분한 용량, 추론 수 ms, 합성 데이터로 라벨링 비용 0 |
| 색 임계처리 (HSV) | 과제의 "AI 모델 활용" 조건 미충족. 조명 변화에 취약 |
| torchvision Faster R-CNN | 추가 의존성 0이지만 추론 ~10배 느림 → fallback |
| Zero-shot (Grounding DINO 등) | "연두 vs 노랑" 미세 색 구분의 신뢰성 불확실 → **§2.5 에서 실측 검증** (mAP50 0.68~0.88, 연두→노랑 혼동 실재) |

**실시간성이 요구사항**이다: 인식이 시뮬레이션 루프 안에서 돌기 때문에 프레임 예산을
먹으면 제어 주기에 영향을 준다. 실측: 단독 2.2ms, Isaac Sim 렌더와 GPU 공유 시 27ms
(12배!) — 이 측정이 발행 주기 10Hz(예산의 27%) 설계의 근거다.

### 2.2 합성 데이터 (Replicator)

train 2,800 + val 300 (**다른 시드**), 박스 12,357개, 라벨링 비용 0.
학습 이미지는 런타임과 **같은 코드(`scene_setup.py`)** 로 만든 씬을 **같은 카메라**로
찍는다 — 도메인 갭이 원리적으로 없다. 랜덤화: 큐브 배치(흩어짐 60% / 부분 탑 40%),
로봇 관절, 조명 세기·색·방향, 카메라 지터(±2cm/±2°), 그리퍼 파지(운반 중) 장면 보충.

결과: **mAP50 0.9949 / mAP50-95 0.9695**, 노랑↔연두 오분류 600건 중 1건.
(연두 색상값은 눈대중이 아니라 렌더 픽셀 실측으로 후보 5종을 비교해 선정)

### 2.3 학습/런타임 환경 분리

```
[학습 venv]      ultralytics + torch 2.7.0  ──학습/export──▶ best.torchscript
[isaacsim 환경]  순정 torch + torchvision.ops.nms  ──로드──▶ 추론 (신규 패키지 0개)
```

ultralytics 의 의존성(opencv GUI 빌드 등)이 Isaac Sim 환경을 깨뜨릴 위험을 차단한다.
YOLOv8 원시 출력(NMS 미적용)을 직접 디코드하며, 배포 전 **decode parity 게이트**로
ultralytics 와의 일치를 증명했다 (33박스 전부 **IoU 1.00000**). torch 는 양쪽 모두
2.7.0 으로 고정 (TorchScript 하위호환 리스크 제거).

### 2.4 3D 위치 추정 — 깊이는 "앞면"을 준다

깊이(`distance_to_image_plane`)는 큐브의 카메라 쪽 표면을 준다. 그대로 쓰면 평균
**28.3mm** 어긋난다(큐브 반변 25.8mm — 파지 실패권). 광선-박스 교점의 정확해
`t = h / max(|uₓ|,|u_y|,|u_z|)` 로 보정하면 **2.4mm** (n=120, p95 4.2mm).
오차 분해: 기하 체인 0.8mm(픽셀 양자화) + 검출기 1.6mm(실루엣 중심 편차).
그림: `media/figures/ablation_depth_correction.png`

### 2.5 확장 — zero-shot 대안 실측 비교 (open-vocabulary / VLM)

§2.1 의 기각 사유를 가정으로 남기지 않고, **학습 0·프롬프트만으로** 검출하는 세 계열을
채택 모델과 **같은 val 300장, 같은 채점기**(`eval/zeroshot/common.py`)로 실측했다.
인식기는 "RGB → 클래스+박스" 생산자 자리 하나이므로(3D 추정·브리지·행동 계층 무수정)
그대로 갈아끼워 비교할 수 있다 — 파이프라인 모듈성의 검증이기도 하다.

| 모델 | 방식 | mAP50 | pick 정확도* | 지연/장 |
|---|---|---|---|---|
| **YOLOv8n fine-tuned (SDG)** ✅ 채택 | 폐쇄셋 학습 | **0.999** | **0.999** | **7 ms** |
| Grounding DINO (tiny) | zero-shot open-vocab | 0.878 | 0.873 | 177 ms |
| Qwen2.5-VL-3B | zero-shot 생성형 VLM | 0.820† | 0.867† | 3,852 ms |
| YOLO-World v2 (s) | zero-shot open-vocab (실시간) | 0.680 | 0.697 | 13 ms |

(\*) pick 정확도 = 브리지 정책(클래스별 최고 신뢰도 1개 픽)이 실제 그 색 큐브에 맞은
프레임 비율 — 스태킹 성공을 가장 직접 예측하는 지표. (†) 장당 ~4s 라 서브셋 30장.

실측이 말해주는 것 (그림 `media/figures/zeroshot_compare.png`):

- **연두가 실제 급소다**: Grounding DINO 는 연두 GT 를 "yellow" 로 **67회** 오분류
  (yellow/green AP 0.77/0.81). §2.1 의 기각 사유가 수치로 확인됐다.
- **프롬프트가 곧 하이퍼파라미터**: YOLO-World 는 연두를 "green cube" 로 부르면
  mAP50 0.525, "light green cube" 로 부르면 0.680 — 학습 대신 프롬프트를 튜닝하게 된다.
- **신뢰도 보정 부재**: zero-shot 계열은 conf 0.25 게이트에서 검출이 거의 전멸한다
  (순위 능력과 별개). 고정 게이트 브리지에 그대로 꽂을 수 없고 게이트 재설계가 필요.
- **생성형 VLM 은 색 판단이 가장 깨끗**(Qwen 혼동 3/120)하지만 박스가 느슨하고
  ~4 s/장 — 10 Hz 제어 루프가 아니라 오프라인 분석용 프로필이다.

**결론 — 대체가 아니라 역할 분담.** 폐쇄셋·10 Hz 루프인 본 과제의 runtime 은 파인튜닝
소형 모델이 맞다. zero-shot 계열의 자리는 (1) 신규 객체 프로토타이핑, (2) **실환경
auto-labeling(교사) → 소형 모델 증류(학생)** 파이프라인이다. 본 과제는 시뮬레이터
GT(Replicator)가 교사 역할을 했지만, GT 가 없는 실환경에서는 zero-shot 이 그 자리를
채운다 — 즉 두 접근은 경쟁 관계가 아니라 한 파이프라인의 두 단계다.

재현 (학습 venv 위에 추가 의존성만 설치, isaacsim 환경 무관):

```bash
training/.venv/bin/pip install -r eval/zeroshot/requirements-extra.txt
cd training
.venv/bin/python ../eval/zeroshot/run_finetuned.py   # 기준선 재채점 (동일 채점기)
.venv/bin/python ../eval/zeroshot/run_yoloworld.py   # + CLIP 가중치 자동 다운로드
.venv/bin/python ../eval/zeroshot/run_gdino.py       # ~700MB 다운로드
.venv/bin/python ../eval/zeroshot/run_qwen.py        # ~7GB 다운로드, 서브셋 30장
.venv/bin/python ../eval/zeroshot/summarize.py       # 비교표 + 그림 생성
```

## 3. 검증 결과

### 3.1 과제 요구 vs 실제 검증 범위

"한 번 되는 것"과 "항상 되는 것"은 다른 문제라고 보고, 과제가 명시한 시나리오를
넘어서는 조건에서 검증했다. 강건성 결함(디버그 마커 오염, 운반 경로 정체, 손목 감김
데드락 등)은 전부 이 스트레스 조건이 끌어냈다 — 명세 조건만 검증했다면 데모/촬영
중에 처음 만났을 결함들이다.

| 항목 | 과제 요구 | 본 구현의 검증 범위 |
|---|---|---|
| 초기 큐브 배치 | 예제 기본 배치 (고정) | **랜덤 위치+yaw** (작업공간 r 0.40~0.75m, yaw 0~90°) |
| belief 초기값 | (제약 없음) | **평균 361mm 의도적 오염 시작** — 인식이 교정해야만 성공 |
| 인식의 역할 | 색상 + 3D 위치 인식 | **ghost belief 구조** — 로봇이 인식 결과만 보고 동작, 인식 오차 = 파지 오차 (인식 없이 성공이 구조적으로 불가능) |
| Orientation | "필요 시 추정" | 랜덤 yaw 로 **필요성을 실측 입증 후** 구현 (정렬 가정 파지가 45° 큐브 모서리를 쳐내는 실패 → 깊이 상면 minAreaRect) |
| 적재 위치 | 사용자 임의 지정 | 사용자 지정 + 유효성 사전 검증(도달성·스폰 간섭) + GUI 입력 |
| 증빙 | 실행 영상/스크린샷 | 영상 + **N=10 배치 평가 정량** (성공률·시간·오차·CSV) |

과제 본연의 시나리오(예제 기본 배치)는 M5 게이트에서 **5회 연속 완주**로 별도 검증했다.
작업공간 r 0.40~0.75m 는 스스로 강화한 랜덤 조건에 부과한 운영 명세이며, 본연 시나리오의
스폰(r 0.5~0.8)은 그 안에 여유 있게 포함된다 — 요구사항을 좁힌 것이 아니다.

### 3.2 게이트별 결과

| 게이트 | 결과 |
|---|---|
| 기하 체인 (GT 픽셀 왕복) | 오차 평균 0.8mm — AI 투입 전 검증 |
| decode parity | IoU 1.00000 (33/33), 클래스 불일치 0 |
| 인식 정확도 (YOLO+3D, 정지 120측정) | 평균 2.4mm / p95 4.2mm / 미검출 0 |
| **End-to-end 스태킹 (기본 위치)** | **5회 연속 완주**, ~35s, 탑 오차 1~8mm |
| End-to-end (대체 위치 0.45,0.25) | 완주 |
| 제약 위반 위치 (0.35,-0.30) | 정상 거부 (스폰과 0.15m 미만 — UI 가 사전 차단하는 입력) |
| **배치 평가 (랜덤 위치+yaw 스폰 10회, belief 오염 시작)** | **10/10 (100%)**, 평균 44.9s, 탑 오차 최대 7.4mm |

배치 평가는 과제 범위를 넘어선 스트레스 테스트다: 매 트라이얼 큐브를 무작위
위치·yaw 로 흩고(파지 유효 작업공간 r 0.40~0.75m), **belief 를 평균 361mm 오염시킨
상태로 시작**한다 — 인식이 스스로 교정해야만 성공하므로 "인식이 진짜 루프 안에 있다"의
정량 증명이다. 초기 성공률 22% 에서 원인 5+3개를 계측으로 규명·수정하며 100% 에 도달했다
(대표 사례: 디버그 마커 자기관측 오염, 베이스 배리어-탑 인력 평형 정체, 배치 자세
고정 선호가 만드는 손목-베이스 감김 데드락 — 상세는 커밋 히스토리).
그림: `media/figures/trials_summary.png`, 원자료: `media/m6/trials.csv`

## 4. 설치와 실행

모든 스크립트는 **저장소 상대 경로**만 사용한다 — 어느 위치에 체크아웃해도 동작하며,
아래 명령은 전부 저장소 루트에서 실행한다고 가정한다.

### 4.0 설치 (클린 환경 기준, 최초 1회)

**요구 환경**: Ubuntu 22.04+, NVIDIA RTX GPU(개발·검증: RTX 4080 SUPER 16GB) +
드라이버 550 이상, 디스크 여유 ~60GB, 최초 실행 시 인터넷(Isaac Sim 에셋 다운로드).

```bash
# 1) Python 3.11 가상환경 + Isaac Sim 5.1 (pip 배포판)
conda create -n isaacsim python=3.11 -y
conda activate isaacsim
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# 2) 추론 런타임 (TorchScript 로드 + NMS 용도. 학습용 ultralytics 는 필요 없다 - §2.3)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python-headless==4.11.0.86

# 3) 저장소 (위치 자유)
git clone <저장소> roboe_block_stacking && cd roboe_block_stacking

# 4) 설치 확인 (GPU/Isaac Sim 없이 도는 단위 테스트)
python eval/test_decode_math.py
```

- **학습된 모델이 저장소에 포함**되어 있다(`models/best.torchscript`) — 바로 실행 가능.
  학습 재현은 §4.3(선택).
- Isaac Sim **최초 실행 시** EULA 동의가 필요하고, ZED-X 등 에셋을 클라우드에서
  내려받으므로 첫 로드는 수 분 걸릴 수 있다.

### 4.1 GUI 예제 (권장)

```bash
conda activate isaacsim
# 최초 1회: user_examples 에 등록 (심링크 + import 한 줄. 재실행해도 중복되지 않음)
UE="$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/exts/isaacsim.examples.interactive/isaacsim/examples/interactive/user_examples"
ln -sfn "$(pwd)/isaac_ext/roboe_block_stacking" "$UE/roboe_block_stacking"
grep -q RoboeBlockStackingExtension "$UE/__init__.py" 2>/dev/null || \
  echo "from isaacsim.examples.interactive.user_examples.roboe_block_stacking import RoboeBlockStackingExtension" >> "$UE/__init__.py"

isaacsim
```
**Window → Examples → Robotics Examples → Custom → ROBOE Block Stacking**
→ `Tower X/Y` 입력 → `LOAD` → `Start`

UI 기능:
- **Perception On/Off** — 끄면 로봇이 기본 위치의 고스트로 간다 (인식 의존성을 눈으로 증명)
- **Randomize Cubes** — 실행 중에도 안전한 라이브 랜덤 재배치 (파지 중/탑 큐브 제외).
  belief 는 건드리지 않으므로 인식이 재검출로 따라잡는 과정이 그대로 보인다
- **YOLO 검출 뷰 창** — 검출기가 보는 이미지 + bbox/score 실시간 (캡처 후 주석이라
  되먹임 불가). 뷰포트 3D 마커는 쓰지 않는다: debug_draw 기하가 ZED 카메라에도
  렌더링되어 검출기를 오염시키는 자기관측 문제를 실측으로 확인하고 제거했다
- **Depth 보정 모드** (none 으로 바꾸면 추정이 3cm 치우침), **Belief 구조**(ghost/direct),
  **Perception 패널**(검출·3D좌표·yaw·지연·bridge 판정 실시간)

### 4.2 GUI 없이 (검증/평가)

```bash
python standalone/run_stacking_perception.py            # 인식 기반 E2E (M5 게이트)
python standalone/run_stacking_perception.py --tower 0.45 0.25
python eval/run_trials.py --trials 10 --record          # 배치 평가 + ZED 시점 영상
python standalone/record_demo.py                        # 데모 영상 재생성 (media/demo/)
python standalone/verify_m1_camera.py                   # 기하 체인 검증
python standalone/verify_m4_perception.py               # 인식 정확도 측정
python standalone/verify_example_reload.py              # 예제 재로드 회귀 테스트
python eval/test_decode_math.py                         # 디코드 좌표 단위 테스트 (GPU 불필요)
python eval/benchmark_detector.py --n 200               # 추론 지연 벤치마크 (*)
```

(*) 벤치마크만 예외적으로 검증 이미지가 필요하다 — 데이터셋은 용량 문제로 저장소에
포함하지 않으므로 §4.3 의 `generate_dataset.py` 를 먼저 실행해야 한다.
나머지 명령은 클린 체크아웃에서 바로 동작한다 (모델 포함, 상대 경로만 사용 — 실증됨).

### 4.3 데이터 생성 + 학습 (재현)

```bash
python sdg/generate_dataset.py --train 2500 --val 300              # isaacsim 환경
python sdg/generate_dataset.py --train 300 --val 0 --start-index 2500 --held-prob 0.9 --seed 777
python -m venv training/.venv && training/.venv/bin/pip install -r training/requirements.txt
training/.venv/bin/python training/train.py                         # 학습 + export
training/.venv/bin/python training/verify_torchscript_decode.py     # 배포 전 게이트
```

## 5. 저장소 구성

| 경로 | 내용 |
|---|---|
| `isaac_ext/roboe_block_stacking/` | Isaac Sim GUI 예제 (씬 + behavior + perception) |
| ├ `behavior/` | Block Stacking decider network (스톡 복사본, 변경점 `[ROBOE]` 주석) |
| ├ `perception/` | `zed_camera` · `detector` · `estimator_3d` · `cortex_bridge` |
| └ `scene_setup.py` | 큐브/조명/고스트/적재 위치 검증 (SDG·런타임 공용) |
| `sdg/` | Replicator 합성 데이터셋 생성 |
| `training/` | YOLO 학습·검증·export (별도 venv) |
| `models/` | best.pt / **best.torchscript**(런타임) / model_meta.json |
| `standalone/` | E2E 러너 및 검증 스크립트 |
| `eval/` | 배치 평가, 단위 테스트, 벤치마크, 그림 생성 |
| `media/` | 스크린샷 · 오버레이 · 영상 · 발표 그림 |

## 6. 명세 해석 (가정 명시)

- **"집기 순서" = 쌓는 순서**: 먼저 집는 빨강이 탑의 맨 아래.
  (반대 해석이면 `behavior/block_stacking_behavior.py` 의 `order_preference` 한 줄로 대응)
- **"연두" ↔ `GreenCube`**: 이름은 behavior 결합 때문에 유지, **색만** 연두로 조정
  (렌더 픽셀 실측으로 노랑과 가장 잘 구분되는 값 선정, 혼동행렬로 최종 검증)
- **적재 위치**: GUI 입력. behavior 에서 역산한 제약(베이스 0.30~0.75m, 큐브 스폰과
  0.15m 초과)을 사전 검증 — 어기면 로봇이 조용히 홈으로 가는 것을 실측으로 확인했다.
- 실물 ZED-X 확장 시: `zed_camera.py` 의 데이터 소스만 ZED SDK 스트림으로 교체하면
  detector / estimator / bridge 는 무수정 (계층 분리의 근거).
