[← README](../README.md)

# ② 파지점 추정 — 바운딩박스 → 3D 위치·자세




> **입력** :  [①](01_perception.md)의 Bounding Box + Depth 
> **출력** :  큐브 중심의 World Position (x, y, z) + Orientation (yaw).





## Position (x,y,z) — 3단계

깊이 카메라가 주는 값은 큐브 **중심이 아니라 카메라 쪽 표면(앞면)까지의 거리**.
그대로 쓰면 반 큐브만큼 어긋남. 그래서:

1. **깊이 샘플링** — Bounding Box 중앙 40% Pixel영역만을 깊이 샘플링 한 후,  그 **중앙값 (median)** 을 선택
2. **역투영** — 3D Point를 x,y,z로 복원 = 큐브 앞면
3. **중심 보정** — 시선 광선 u 와 정육면체(반변 h)의 교점 공식으로 앞면→중심 이동:
   `t = h / max(|uₓ|, |u_y|, |u_z|)`, `중심 = 앞면 + t·u`. 마지막에 바닥 관통 방지 z 클램프

| | 위치 오차 (정지 큐브 120회 측정) |
|---|---|
| 보정 없이 앞면 좌표 사용 | 평균 **28.3mm** — 파지 실패권 (반큐브 25.8mm 초과) |
| 보정 적용 | 평균 **2.4mm**, 95백분위 4.2mm |






## Orientation (yaw) — 상면만 투영해서 직사각형 피팅

큐브가 회전한 채 놓이면 "정렬돼 있다" 가정 파지는 모서리를 침 (랜덤 배치에서 실측).

1. 박스 안의 깊이 픽셀들을 전부 3D 로 역투영
2. 최고점 기준 **상면 1.5cm 슬라이스**만 남김 (옆면 픽셀 배제)
3. 바닥(xy)에 투영 → 최소 외접 사각형(`cv2.minAreaRect`)의 기울기 = yaw
4. 정육면체는 90° 대칭 → 0~90° 로 정규화

yaw 는 [③](03_decision_control.md)에서 "등가 파지 자세 4개 중 손목 회전 최소" 선택에 쓰임.

**코드**: `perception/estimator_3d.py` · 정확도 측정: `standalone/verify_m4_perception.py` ·
보정 전후 비교 그림: `media/figures/ablation_depth_correction.png`

**다음**: [③ 판단·제어](03_decision_control.md)
