[← README](../README.md)

# ① AI 인식 — RGB 에서 색상 큐브 검출

> **입력** :  ZED-X RGB 프레임 
> **출력** : 색상별 큐브의 클래스 + 바운딩박스 + 신뢰도.



## 기본 모델 — YOLOv8n 파인튜닝

| 후보                 | 판단                                      |
| ------------------ | --------------------------------------- |
| **YOLOv8n 파인튜닝** ✅ | 4클래스·고정 시점에 충분한 용량, 추론 수 ms, 라벨링 비용 0   |
| 색 임계처리 (HSV)       | 과제의 "AI 모델 활용" 조건 미충족, 조명 변화에 취약        |
| Faster R-CNN       | 추론 ~10배 느림                              |
| zero-shot 계열       | 학습 불필요가 장점 → 4종 실측 비교로 검증 (README 결과 표) |


## 학습 데이터 — Replicator를 활용한 Labeling

Isaac Sim **Replicator**(렌더링하면서 정답 박스·클래스까지 자동 생성하는 도구)로
런타임과 **같은 씬을 같은 카메라**로 찍어 생성 

- 큐브 위치·회전, 로봇 자세, 조명, 카메라 미세 zitter 적용
- train 2,800장 / val 300장
- 결과: **mAP50 0.9949**, 노랑↔연두 오분류 600건 중 1건




## 배포 — Isaac Sim 환경에 추가 패키지 0개

학습은 별도 conda (ultralytics), 런타임은 **TorchScript** 로 내보내 순정 torch 만으로 로드.
내보낸 모델과 원본의 출력 일치를 배포 전 검증 (박스 33개 전부 IoU 1.00000).



## AI Source

검출기는 파이프라인에서 "RGB→박스" 생산자 자리 하나 — 이 지점만 갈아끼우면
나머지(②③④)는 무수정. GUI 드롭다운으로 실행 중 전환됨.

| 소스             | 로드 방식                                      | Threshold      |
| -------------- | ------------------------------------------ | -------------- |
| YOLOv8n 파인튜닝   | TorchScript in-process                     | 0.5            |
| YOLO-World v2  | TorchScript in-process (프롬프트 임베딩을 미리 bake) | 0.003*         |
| Grounding DINO | 학습 venv 서브프로세스 워커 (비동기)                    | 0.25           |
| Qwen2.5-VL     | 〃                                          | (게이트 무의미 — 아래) |

(\*) zero-shot 신뢰도는 보정이 안 되어 있음 (YOLO-World 정검출 신뢰도 중앙값 0.0105).
그래서 Threshold 를 모델별로 자동 전환함. Qwen 은 생성형이라 신뢰도 개념이 없어
고정 0.99 로 출력되고, 보호는 [③](03_decision_control.md)의 작업공간/동결 게이트가 담당.

느린 워커(GDINO 177ms, Qwen ~4s)는 비동기 mailbox 구조. 워커가 놀고 있을 때만
프레임을 받고 항상 가장 최근 결과를 반환하므로 시뮬 루프를 막지 않음. VRAM 은
순차 점유 (전환 시 이전 워커를 먼저 종료. 16GB 에서 동시 점유 시 OOM 실측).

**코드**: `perception/detector.py` (디코드+NMS) · `perception/detector_hub.py` (4종 전환) ·
`sdg/generate_dataset.py` (데이터 생성) · `training/train.py` (학습+export)

**다음**: [② 파지점 추정](02_grasp_point.md)
