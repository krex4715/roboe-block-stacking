"""[ROBOE] 2D 픽셀 + 깊이 -> 3D 월드좌표 추정.

핵심 문제: 깊이 센서가 주는 건 큐브의 **앞면(카메라를 향한 표면)** 좌표이지 중심이 아니다.
그대로 쓰면 큐브 중심이 카메라 쪽으로 치우쳐 계산되어 파지 위치가 어긋난다.

보정 방법 3가지를 모두 구현해 두고 실험으로 비교한다 (ablation):
  "none" : 보정 없음 - 표면점을 그대로 중심으로 사용 (베이스라인)
  "ray"  : 시선 방향으로 반큐브(half)만큼 밀기 - 카메라가 면을 정면으로 볼 때만 정확
  "box"  : 광선-박스 기하를 정확히 푼 해 (권장, 축정렬 큐브 가정)

"box" 유도:
  중심이 원점이고 한 변이 2h인 축정렬 정육면체를 생각한다. 중심을 지나는 시선 방향 u(단위벡터)를
  거꾸로 따라가면 표면과 만나는 지점은 max(|t·u_x|, |t·u_y|, |t·u_z|) = h 를 만족하는 t이다.
      t = h / max(|u_x|, |u_y|, |u_z|)
  즉 표면점에서 시선 방향으로 t 만큼 진행하면 정확히 중심이다.
  - 면을 정면으로 보면 max|u_i| = 1  -> t = h        ("ray" 방식과 동일)
  - 모서리 방향으로 보면 max|u_i| = 1/sqrt(3) -> t = 1.732h  ("ray"는 이때 최대로 틀린다)
"""

import numpy as np

# Cortex 예제의 큐브 한 변 길이 (block_stacking_behavior.py 의 block_height 와 동일해야 함)
CUBE_SIZE = 0.0515
CUBE_HALF = CUBE_SIZE / 2.0


def backproject_pixels(camera, pixels, depths):
    """픽셀 좌표 + 깊이 -> 월드 좌표(표면점).

    Args:
        camera: isaacsim.sensors.camera.Camera (pinhole 이어야 함)
        pixels: (n, 2) [u(가로, col), v(세로, row)]
        depths: (n,) `distance_to_image_plane` 값.
                주의: 광선 거리가 아니라 **이미지 평면까지의 수직 거리(Z_cam)** 여야 한다.
                Isaac Sim 역투영이 K^-1 @ [u*d, v*d, d] 로 계산하므로 Z_cam = d 가 되기 때문.
    Returns:
        (n, 3) 월드 좌표 (큐브 앞면 위의 점)
    """
    pixels = np.asarray(pixels, dtype=np.float32).reshape(-1, 2)
    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    return np.asarray(camera.get_world_points_from_image_coords(pixels, depths), dtype=np.float64)


def surface_to_center(points_surface, camera_position, half=CUBE_HALF, mode="box"):
    """표면점 -> 큐브 중심 (위 docstring의 보정 3종)."""
    p = np.atleast_2d(np.asarray(points_surface, dtype=np.float64))
    cam = np.asarray(camera_position, dtype=np.float64).reshape(3)

    if mode == "none":
        return p.copy()

    rays = p - cam                                    # 카메라 -> 표면점
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    u = rays / np.maximum(norms, 1e-9)                # 단위 시선벡터

    if mode == "ray":
        t = np.full((len(p), 1), half)
    elif mode == "box":
        # 축정렬 큐브 가정. 큐브가 z축으로 회전(yaw)해 있으면 yaw로 u를 회전시켜 넣으면 된다.
        t = half / np.max(np.abs(u), axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown mode: {mode}")

    return p + t * u


def clamp_to_support(points, half=CUBE_HALF, support_z=0.0):
    """큐브 중심이 바닥면 아래로 내려가지 않도록 z를 클램프.

    깊이 노이즈로 중심이 테이블 아래로 추정되면, 그 값을 belief에 반영하는 순간
    물리 엔진이 큐브를 지면 밖으로 튕겨낸다(관통 팝). 그걸 막는 안전장치.
    """
    p = np.atleast_2d(np.asarray(points, dtype=np.float64)).copy()
    p[:, 2] = np.maximum(p[:, 2], support_z + half)
    return p


def sample_depth(depth_image, pixel, window_ratio=0.4, box=None):
    """bbox 중앙 영역의 깊이 **중앙값**을 뽑는다.

    단일 픽셀 값을 쓰면 경계에서 배경 깊이를 물기 쉽다. bbox 중앙부만 보고 중앙값을 취하면
    배경 침범·노이즈에 강해진다. box가 None이면 픽셀 주변 소형 윈도우를 사용한다.

    Returns:
        float 깊이. 유효 값이 없으면 None.
    """
    h, w = depth_image.shape[:2]
    u, v = float(pixel[0]), float(pixel[1])

    if box is not None:
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        hw = (x1 - x0) * window_ratio / 2.0
        hh = (y1 - y0) * window_ratio / 2.0
        u0, u1 = int(round(cx - hw)), int(round(cx + hw))
        v0, v1 = int(round(cy - hh)), int(round(cy + hh))
    else:
        r = 2
        u0, u1, v0, v1 = int(round(u)) - r, int(round(u)) + r, int(round(v)) - r, int(round(v)) + r

    u0, v0 = max(0, u0), max(0, v0)
    u1, v1 = min(w - 1, u1), min(h - 1, v1)
    if u1 < u0 or v1 < v0:
        return None

    patch = depth_image[v0 : v1 + 1, u0 : u1 + 1].astype(np.float32)
    valid = patch[np.isfinite(patch) & (patch > 1e-4)]
    if valid.size == 0:
        return None
    return float(np.median(valid))
