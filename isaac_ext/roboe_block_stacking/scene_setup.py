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
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.prims import create_prim

# 베이스 예제(franka_cortex.py)와 동일한 큐브 기하
CUBE_SIZE = 0.0515
CUBE_HALF = CUBE_SIZE / 2.0
CUBE_Y = -0.4
CUBE_X_RANGE = (0.3, 0.7)

# 이름은 behavior의 desired_stack("<Color>Cube")과 일치해야 한다.
# 색은 베이스 예제 값. (M2에서 GreenCube를 과제의 '연두'로 조정 예정)
CUBE_SPECS = [
    ("RedCube", (0.7, 0.0, 0.0)),
    ("BlueCube", (0.0, 0.0, 0.7)),
    ("YellowCube", (0.7, 0.7, 0.0)),
    ("GreenCube", (0.0, 0.7, 0.0)),
]


def default_cube_positions():
    """베이스 예제의 초기 배치: x는 균등 분포, y는 고정, z는 반큐브(지면 접촉)."""
    xs = np.linspace(CUBE_X_RANGE[0], CUBE_X_RANGE[1], len(CUBE_SPECS))
    return {name: np.array([x, CUBE_Y, CUBE_HALF]) for x, (name, _) in zip(xs, CUBE_SPECS)}


def add_cubes(scene, positions=None):
    """큐브 4개를 씬에 추가하고 {이름: 객체} 를 반환."""
    positions = positions or default_cube_positions()
    cubes = {}
    for name, color in CUBE_SPECS:
        cubes[name] = scene.add(
            DynamicCuboid(
                prim_path=f"/World/Obs/{name}",
                name=name,
                size=CUBE_SIZE,
                color=np.array(color),
                position=np.asarray(positions[name]),
            )
        )
    return cubes


def add_lighting(dome_intensity: float = 900.0, distant_intensity: float = 1200.0):
    """돔(전역 확산) + 디스턴트(방향성 그림자) 2등 구성.

    돔만 있으면 그림자가 없어 큐브 경계가 흐려지고, 디스턴트만 있으면 그늘이 새까매져
    한쪽 면이 사라진다. 둘을 섞어야 색과 형태가 모두 살아난다.

    세기는 흰색 Franka가 과노출(픽셀 포화)되지 않는 선에서 정했다. 포화되면 로봇 표면의
    형태 정보가 사라지고, 자동 노출이 있는 실제 카메라와도 다른 그림이 된다.
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
