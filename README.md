# [Take-home task] AI Vision 기반 Block Stacking

Isaac Sim 5.1 환경에서 **ZED-X RGB-D 카메라**와 **AI 검출 모델(YOLOv8n)** 로 색상 큐브를 인식하고,
Franka 로봇 암이 **빨강 → 노랑 → 연두 → 파랑** 순서로 사용자 지정 위치에 쌓는 파이프라인.

> 베이스: Isaac Sim 내장 **Franka Cortex Examples** (Block Stacking behavior, Apache-2.0)

*(작성 중 — 정량 결과와 실행 영상은 완료 후 채워집니다)*

---

## 1. 전체 구조 (Vision → Perception → Manipulation)

```
[Isaac Sim 씬]
   ZED-X (고정, Translate 1.0/0.0/1.0, Orient 0.0/-50.0/180.0, Kinematic)
        │
        ├── RGB  ──────▶ YOLOv8n (TorchScript, in-process) ──▶ bbox + 클래스
        │                                                          │
        └── Depth ─────────────────────────────────────────────────┤
                        (distance_to_image_plane)                   ▼
                                                       3D 추정기 (estimator_3d)
                                              깊이 중앙값 → 역투영 → 큐브 중심 보정
                                                                    │ (position, orientation)
                                                                    ▼
                                                     Cortex Bridge (게이팅·필터)
                                                     CortexObject.set_measured_pose()
                                                                    │
                                                                    ▼
                                            Block Stacking Decider Network (Cortex)
                                                                    │
                                                                    ▼
                                              MotionCommander (RMPFlow) → Franka
```

**설계의 핵심**: Cortex 프레임워크는 원래 *belief(로봇이 믿는 세계)* 와 *measured(인식 결과)* 를
분리하도록 설계돼 있고, `CortexObject.set_measured_pose()` 라는 인식 주입 지점을 갖고 있다.
다만 기본 설치본에는 이 API를 호출하는 코드가 한 곳도 없어, 예제는 시뮬레이터의 ground truth를
그대로 읽는다. **본 구현은 바로 그 지점을 카메라 기반 AI 인식으로 채운다** — 의사결정 로직
(어떤 블록을 다음에 집을지, 어디에 놓을지)은 손대지 않고 *정보의 출처*만 교체한다.

## 2. 인식 방식 / 모델 선택 이유

### 2.1 왜 학습 기반 검출기(YOLO)인가

| 후보 | 판단 |
|---|---|
| **YOLOv8n 파인튜닝** ✅ 채택 | 4클래스·고정 시점 문제에 충분한 용량, 추론 수 ms, 합성 데이터로 라벨링 비용 0 |
| 색 임계처리 (HSV) | 과제의 "AI 모델 활용" 조건 미충족. 조명 변화에 취약. 비교 베이스라인으로만 가치 |
| torchvision Faster R-CNN | 추가 의존성 0이지만 추론 ~10배 느림 → fallback 으로만 유지 |
| Zero-shot (Grounding DINO 등) | 학습 불필요하나 "연두 vs 노랑" 같은 미세 색 구분의 신뢰성이 불확실 |

**실시간성이 요구사항**인 점이 결정적이었다. 인식이 시뮬레이션 루프 안에서 돌기 때문에
프레임 예산을 많이 먹으면 시뮬이 느려지고 로봇 제어 주기에도 영향을 준다.

### 2.2 왜 합성 데이터인가

1. **라벨링 비용 0** — Replicator가 정답 bbox를 자동 생성
2. **도메인 갭이 원리적으로 없다** — 배포 대상이 시뮬레이터 자체다.
   학습 이미지를 런타임과 **같은 코드(`scene_setup.py`)** 로 만든 씬을
   **같은 카메라(ZED-X 좌안, 명세 pose)** 로 찍어 생성한다
3. 도메인 랜덤화(조명·로봇 자세·큐브 배치·카메라 지터)로 강인성 확보

### 2.3 왜 학습 환경과 런타임 환경을 분리했나

```
[학습 venv]      ultralytics + torch 2.7.0  ──학습/export──▶ best.torchscript
[isaacsim 환경]  순정 torch + torchvision.ops.nms  ──로드──▶ 추론
```

ultralytics는 `opencv-python`(GUI 빌드)·matplotlib 등을 끌고 오는데, Isaac Sim 환경은
`opencv-python-headless`/numpy 1.26에 고정돼 있어 충돌 위험이 있다. 시뮬레이터가 안 뜨면
과제 전체가 막히므로 **Isaac Sim 환경에는 신규 패키지를 하나도 설치하지 않고**,
TorchScript 산출물만 넘겨 순정 torch로 추론한다.

대신 YOLOv8의 원시 출력(NMS 미적용)을 직접 디코드해야 하는데, 이 디코드가 ultralytics와
동일한 결과를 내는지 **배포 전에 반드시 검증**한다 (`training/verify_torchscript_decode.py`).

> torch 버전은 양쪽 모두 **2.7.0+cu128** 로 고정했다. TorchScript 아카이브는 만든 버전보다
> 낮은 런타임에서 로드가 실패할 수 있어, 소비하는 쪽(Isaac Sim)에 생산하는 쪽을 맞췄다.

### 2.4 3D 위치 추정 — 깊이는 "앞면"을 준다

깊이 센서가 주는 값은 큐브의 **카메라를 향한 표면** 좌표이지 중심이 아니다. 그대로 쓰면
큐브 중심이 카메라 쪽으로 치우쳐 계산된다 (실측 평균 오차 **31mm** — 큐브 한 변이 51.5mm이니
파지 실패로 이어질 수 있는 크기).

중심이 원점이고 한 변이 2h인 축정렬 정육면체에서, 중심을 지나는 시선 단위벡터 `u` 를 따라
표면까지의 거리는 `t = h / max(|uₓ|, |u_y|, |u_z|)` 이다. 이 값만큼 표면점을 시선 방향으로
밀면 정확히 중심이 된다.

| 보정 방식 | 평균 오차 |
|---|---|
| 보정 없음 (표면점 그대로) | 31.1 mm |
| 반큐브만큼 밀기 | 5.4 mm |
| **광선-박스 정확해** ✅ | **0.8 mm** |

남은 0.8mm는 픽셀 양자화(1.15m 거리에서 1픽셀 ≈ 2.3mm)로 설명되는 수준이다.

## 3. 실행 방법

### 3.1 GUI 예제 (권장)

```bash
conda activate isaacsim
# 최초 1회: 예제를 Isaac Sim 에 등록
UE=$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/exts/isaacsim.examples.interactive/isaacsim/examples/interactive/user_examples
ln -sfn "$PWD/isaac_ext/roboe_block_stacking" "$UE/roboe_block_stacking"
echo "from isaacsim.examples.interactive.user_examples.roboe_block_stacking import RoboeBlockStackingExtension" >> "$UE/__init__.py"

isaacsim
```
**Window → Examples → Robotics Examples → Custom → ROBOE Block Stacking**
→ `Tower X/Y` 입력 → `LOAD` → `Start`

### 3.2 GUI 없이 (배치 평가/검증)

```bash
python standalone/run_stacking.py --tower 0.45 0.25   # 스태킹 실행 + 자동 성공 판정
python standalone/verify_m1_camera.py                 # 카메라·3D 역투영 검증
python standalone/verify_example_reload.py            # 예제 재로드 회귀 테스트
python eval/test_decode_math.py                       # 디코드 좌표 수학 단위 테스트
```

### 3.3 데이터 생성 + 학습 (재현용)

```bash
python sdg/generate_dataset.py --train 2500 --val 300          # isaacsim 환경
python -m venv training/.venv && training/.venv/bin/pip install -r training/requirements.txt
training/.venv/bin/python training/train.py                     # 학습 + export
training/.venv/bin/python training/verify_torchscript_decode.py # 배포 전 게이트
```

## 4. 저장소 구성

| 경로 | 내용 |
|---|---|
| `isaac_ext/roboe_block_stacking/` | Isaac Sim GUI 예제 (씬 + behavior + perception) |
| ├ `behavior/` | Block Stacking decider network (스톡 복사본, 변경점은 `[ROBOE]` 주석) |
| ├ `perception/` | `zed_camera` · `detector` · `estimator_3d` · `cortex_bridge` |
| └ `scene_setup.py` | 큐브 스펙·조명·적재 위치 검증 (SDG/런타임 공용) |
| `sdg/` | Replicator 합성 데이터셋 생성 |
| `training/` | YOLO 학습·검증·export (별도 venv) |
| `models/` | 학습된 가중치 (pt / torchscript / onnx) + `model_meta.json` |
| `standalone/` | GUI 없이 실행하는 러너 및 검증 스크립트 |
| `eval/` | 정량 평가 (3D 오차, 성공률, 디코드 단위 테스트) |
| `media/` | 실행 영상·스크린샷 |

## 5. 명세 해석 (가정 명시)

- **"집기 순서" = 쌓는 순서로 해석**: 먼저 집는 빨강이 탑의 맨 아래.
  (반대 해석이라면 `behavior/block_stacking_behavior.py` 의 `order_preference` 한 줄로 대응 가능)
- **"연두" ↔ `GreenCube`**: 예제 큐브 이름은 behavior의 `desired_stack` 과 결합돼 있어 유지하고,
  **색만** 연두(yellow-green)로 조정했다. 색상값은 눈대중이 아니라 렌더 픽셀 실측으로 선정했다
  (노랑과 가장 잘 구분되는 지점).
- **적재 위치**: 사용자가 GUI에서 입력. behavior 코드에서 역산한 제약(로봇 베이스에서 0.30~0.75m,
  큐브 스폰과 0.15m 초과)을 만족하는지 검증 후 적용한다.
