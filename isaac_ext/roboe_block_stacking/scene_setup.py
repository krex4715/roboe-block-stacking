"""[ROBOE] 씬 구성 공통 모듈 (큐브 스펙 + 조명).

**왜 공통 모듈로 빼는가**: 이 씬은 세 곳에서 만들어진다.
    1) 합성 데이터 생성 (sdg/generate_dataset.py)   - YOLO를 학습시킬 이미지
    2) GUI 예제 런타임 (roboe_stacking_example.py)  - YOLO를 실제로 돌릴 이미지
    3) standalone 러너/검증 스크립트
세 곳의 외형(특히 조명과 큐브 색)이 다르면 학습-배포 도메인 갭이 생겨 검출 성능이 떨어진다.
스펙을 한 곳에 두고 공유해서 그 갭을 원천 차단한다.

조명을 명시적으로 넣는 이유: World를 새로 만들면 기본 스테이지가 거의 어둡고, GUI는 자체
기본 조명을 갖는다. 즉 조명을 명시하지 않으면 실행 경로마다 밝기가 달라진다.
"""

import numpy as np
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.utils.prims import create_prim
from pxr import UsdGeom

# 베이스 예제(franka_cortex.py)와 동일한 큐브 기하
CUBE_SIZE = 0.0515
CUBE_HALF = CUBE_SIZE / 2.0
CUBE_Y = -0.4
CUBE_X_RANGE = (0.3, 0.7)

# 이름은 behavior의 desired_stack("<Color>Cube")과 일치해야 하므로 **이름은 바꾸지 않는다**.
# 색만 과제 명세에 맞춘다: GreenCube 를 초록(0,0.7,0) -> **연두(yellow-green)** 로 조정.
#
# 연두 알베도는 눈대중이 아니라 **렌더 픽셀 실측**으로 골랐다. 연두는 노랑과 색상환에서
# 가까워 오분류 위험이 있는데, 알베도를 노랑에서 멀리 둔다고 렌더 색이 멀어지지 않기 때문이다
# (R이 내려가면서 B가 올라가 유클리드 거리가 상쇄된다). 후보 5종 실측 (노랑과의 RGB 거리):
#     (0.55,0.85,0.15) hue 72도 sat  93 -> 82      (0.10,0.80,0.02) hue 96도 sat 174 -> 95  <= 채택
#     (0.45,0.80,0.10) hue 74도 sat 110 -> 70      (0.00,0.70,0.00) hue120도 sat 206 -> 191 (스톡 초록)
#     (0.25,0.85,0.05) hue 82도 sat 143 -> 60 (최악)
# 즉 "연두답게 보이면서 노랑과 가장 잘 구분되는" 지점이 (0.10, 0.80, 0.02) 이다.
# 그래도 다른 색 쌍(193~305)보다는 가까우므로, 혼동 여부를 M3 혼동행렬로 최종 검증한다.
CUBE_SPECS = [
    ("RedCube", (0.7, 0.0, 0.0)),
    ("BlueCube", (0.0, 0.0, 0.7)),
    ("YellowCube", (0.7, 0.7, 0.0)),
    ("GreenCube", (0.10, 0.80, 0.02)),  # 연두 (렌더 hue 96도, sat 174)
]

# 적재 목표 위치 제약 (behavior 코드에서 역산한 값)
#   - 팔 도달 범위: block 픽업이 |p| < 0.25 또는 > 0.81 이면 거부됨 -> 여유를 두고 0.30~0.75
#   - 큐브 스폰과의 거리: 타워에서 0.15m 안쪽 블록은 픽업이 거부됨 -> 0.15m 초과 요구
TOWER_REACH_MIN = 0.30
TOWER_REACH_MAX = 0.75
TOWER_CLEARANCE = 0.15


def default_cube_positions():
    """베이스 예제의 초기 배치: x는 균등 분포, y는 고정, z는 반큐브(지면 접촉)."""
    xs = np.linspace(CUBE_X_RANGE[0], CUBE_X_RANGE[1], len(CUBE_SPECS))
    return {name: np.array([x, CUBE_Y, CUBE_HALF]) for x, (name, _) in zip(xs, CUBE_SPECS)}


def add_cubes(scene, positions=None, name_suffix=""):
    """물리 큐브 4개를 씬에 추가하고 {기본이름: 객체} 를 반환.

    prim path 는 항상 `/World/Obs/<이름>` 으로 고정한다 (SDG 라벨·평가 스크립트가 이 경로를 쓴다).
    `name_suffix` 는 **씬 레지스트리 이름**만 바꾼다 — ghost 모드에서 고스트가 behavior 가 찾는
    이름("RedCube" 등)을 가져가야 하므로 물리 큐브는 "RedCubeReal" 로 비켜준다.
    반환 딕셔너리의 키는 접미사와 무관하게 기본 이름이라, 호출부 코드는 바뀌지 않는다.
    """
    positions = positions or default_cube_positions()
    cubes = {}
    for name, color in CUBE_SPECS:
        cubes[name] = scene.add(
            DynamicCuboid(
                prim_path=f"/World/Obs/{name}",
                name=name + name_suffix,
                size=CUBE_SIZE,
                color=np.array(color),
                position=np.asarray(positions[name]),
            )
        )
    return cubes


def add_belief_cubes(scene, positions=None):
    """로봇이 '믿는' 큐브(고스트) 4개. 물리도 충돌도 없고 **렌더링되지 않는다**.

    왜 필요한가 - 이게 이 과제의 핵심 구조다:
      물리 큐브를 그대로 behavior 에 등록하면, 로봇이 보는 위치 = 실제 위치가 되어
      **인식을 꺼도 완벽하게 동작한다.** 즉 인식이 파이프라인에 실제로 기여하는지 증명할 수 없다.
      고스트를 두면 로봇은 고스트(=인식 결과)만 보고 계획하고, 물리적으로는 진짜 큐브를 집는다.
      -> 인식 오차가 곧 파지 오차가 되고, 카메라를 가리면 로봇이 옛 위치로 간다.
      Cortex 원설계(cortex_ros 의 belief/real 분리)와도 일치한다.

    **렌더링을 끄는 이유**: 고스트가 화면에 보이면 카메라가 고스트를 검출해버려
    인식이 자기 자신을 보는 순환이 된다. 사람이 볼 수 있게는 debug_draw 로 따로 그린다.
    """
    positions = positions or default_cube_positions()
    ghosts = {}
    for name, color in CUBE_SPECS:
        ghost = scene.add(
            VisualCuboid(
                prim_path=f"/World/Belief/{name}",
                name=name,  # behavior 의 desired_stack 이 찾는 이름
                size=CUBE_SIZE,
                color=np.array(color),
                position=np.asarray(positions[name]),
            )
        )
        UsdGeom.Imageable(ghost.prim).MakeInvisible()
        ghosts[name] = ghost
    return ghosts


def add_lighting(dome_intensity: float = 500.0, distant_intensity: float = 700.0):
    """돔(전역 확산) + 디스턴트(방향성 그림자) 2등 구성.

    돔만 있으면 그림자가 없어 큐브 경계가 흐려지고, 디스턴트만 있으면 그늘이 새까매져
    한쪽 면이 사라진다. 둘을 섞어야 색과 형태가 모두 살아난다.

    세기는 **채널 클리핑이 나지 않는 선**에서 정했다. 이게 생각보다 중요하다:
    조명이 세면 큐브의 R/G 채널이 255 근처로 포화되면서 **색상(hue) 정보가 뭉개진다**.
    실측 예 - 돔900/디스턴트1200 에서 연두 큐브가 RGB(237,244,193), 채도 54까지 바래
    노랑(채도 171)과의 RGB 거리가 113밖에 안 됐다(다른 색 쌍은 193~297).
    노랑/연두처럼 색상환에서 가까운 쌍이 있을 때 이건 바로 오분류로 이어진다.
    -> verify_m1_camera.py 가 매 실행마다 색 분리도를 측정해 회귀를 잡는다.

    SDG 단계에서는 이 값들을 랜덤화해 조명 변화에 강인한 검출기를 만든다.
    """
    create_prim(
        prim_path="/World/Lights/DomeLight",
        prim_type="DomeLight",
        attributes={"inputs:intensity": dome_intensity, "inputs:texture:format": "latlong"},
    )
    create_prim(
        prim_path="/World/Lights/DistantLight",
        prim_type="DistantLight",
        # 살짝 기울여 큐브 옆면에 그림자가 생기도록 (위에서 수직으로 쏘면 옆면이 균일해진다)
        orientation=np.array([0.9239, 0.0, 0.3827, 0.0]),  # y축 기준 45도
        attributes={"inputs:intensity": distant_intensity, "inputs:angle": 1.0},
    )


def validate_tower_position(position, cube_positions=None):
    """적재 목표 위치가 물리적으로 가능한지 검사. (ok: bool, message: str) 반환.

    behavior 는 조건을 어기면 조용히 GoHome 으로 빠져 '아무것도 안 하는' 것처럼 보인다.
    사용자가 위치를 입력하는 UI에서는 그 전에 걸러서 이유를 알려주는 편이 낫다.
    """
    p = np.asarray(position, dtype=float)
    r = float(np.linalg.norm(p[:2]))
    if r < TOWER_REACH_MIN:
        return False, f"로봇 베이스에 너무 가깝습니다 (거리 {r:.2f}m < {TOWER_REACH_MIN}m)"
    if r > TOWER_REACH_MAX:
        return False, f"팔이 닿지 않습니다 (거리 {r:.2f}m > {TOWER_REACH_MAX}m)"

    cube_positions = cube_positions or default_cube_positions()
    for name, cp in cube_positions.items():
        d = float(np.linalg.norm(np.asarray(cp)[:2] - p[:2]))
        if d <= TOWER_CLEARANCE:
            return False, f"{name} 스폰 위치와 너무 가깝습니다 ({d:.2f}m <= {TOWER_CLEARANCE}m)"
    return True, f"OK (베이스에서 {r:.2f}m)"


def add_tower_marker(position, scene=None):
    """적재 목표 위치를 눈으로 보이게 하는 얇은 표식판.

    - 물리 없는 VisualCuboid 이고 **register_obstacle 하지 않는다**.
      등록하면 behavior 가 이것도 '쌓을 블록'으로 오인한다.
    - 색은 무채색으로 둬서 검출기가 큐브로 헷갈리지 않게 한다
      (SDG 이미지에도 함께 들어가므로 배경으로 학습된다).
    """
    p = np.asarray(position, dtype=float).copy()
    p[2] = 0.001  # 지면에 살짝 띄워 z-fighting 방지
    marker = VisualCuboid(
        prim_path="/World/TowerMarker",
        name="tower_marker",
        position=p,
        scale=np.array([0.12, 0.12, 0.002]),
        color=np.array([0.25, 0.25, 0.25]),
    )
    return scene.add(marker) if scene is not None else marker
