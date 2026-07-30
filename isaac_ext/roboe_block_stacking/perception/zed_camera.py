"""[ROBOE] ZED-X RGB-D 카메라 래퍼.

과제 명세(PDF)를 코드로 그대로 재현한다:
    카메라 모델 : ZED-X
    설치 위치   : Translate (1.0, 0.0, 1.0) / Orient (0.0, -50.0, 180.0)
    Kinematic Enabled : Check

--- 에셋 구조 (실측으로 확인, 2026-07-30) ---------------------------------
ZED_X.usdc 안에는 depth sensor 템플릿(RenderProduct + OmniSensorDepthSensorSingleViewAPI)이
**존재하지 않는다**. 실제 내용물은 다음과 같다:

    /ZED_X                              Xform  [PhysicsRigidBodyAPI, PhysxRigidBodyAPI]  <- Kinematic 체크 대상
    /ZED_X/base_link/ZED_X/CameraLeft   Camera   <- 기준 카메라로 사용
    /ZED_X/base_link/ZED_X/CameraRight  Camera   <- 스테레오 짝 (현재 미사용)
    /ZED_X/base_link/ZED_X/Imu_Sensor   IsaacImuSensor
    ... Meshes / Looks

따라서 GUI 메뉴(Create > Sensors > ... > ZED_X)가 호출하는 SingleViewDepthSensorAsset 는
이 에셋에 대해 "참조 추가 + 배치"만 하고 depth sensor 는 하나도 만들지 않는다
(templates 딕셔너리가 비어 initialize() 가 no-op). 그래서 여기서는 같은 일을 하는
add_reference_to_stage 를 직접 쓰고, CameraLeft 를 표준 Camera 로 감싸
필요한 annotator(rgb, distance_to_image_plane)를 붙인다. GUI 경로와 결과는 동일하다.

깊이의 출처: 실물 ZED-X는 좌/우 스테레오로 온디바이스 깊이를 계산한다. 시뮬레이터에서는
같은 좌측 광학 프레임 기준의 렌더 깊이(distance_to_image_plane)를 깊이 채널로 사용한다.
(현실적 스테레오 깊이 모델은 SingleViewDepthSensor 로 별도 실험 가능 - 40 실험 노트 참고)

--- 회전 규약: 왜 rotateZYX 인가 (실측으로 확정, 2026-07-30) ----------------
ZED_X 에셋 루트는 원래 xformOp:rotateZYX(double3) 를 갖고 있다
(xformOpOrder = [translate, rotateZYX, scale]). GUI Property 패널의 "Orient" 위젯이
편집하는 대상이 바로 이 op이므로, PDF의 Orient (0,-50,180)은 rotateZYX 값이다.

같은 숫자를 rotateXYZ 로 쓰면 결과가 완전히 달라진다 (USD rotateXYZ 는 열벡터 규약에서
Rz·Ry·Rx 로 합성되기 때문). 실측 결과:
    rotateXYZ(0,-50,180) -> 광축 (-0.643, 0, +0.766)  하늘을 봄. 유한 깊이 0%, 큐브 0/4
    rotateZYX(0,-50,180) -> 광축 (-0.643, 0, -0.766)  테이블을 봄. 유한 깊이 100%, 큐브 4/4 + 타워
또한 이 에셋은 리그 +X 축이 광축인 ROS 바디 규약으로 저작돼 있다
(Z_cam=+X_rig, X_cam=-Y_rig, Y_cam=-Z_rig).
"""

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera

try:  # 설치 형태에 따라 두 경로 모두 존재
    from isaacsim.storage.native import get_assets_root_path
except ImportError:  # pragma: no cover
    from isaacsim.core.utils.nucleus import get_assets_root_path

# --- 과제 명세 상수 -------------------------------------------------------
ZED_X_USD_SUBPATH = "/Isaac/Sensors/Stereolabs/ZED_X/ZED_X.usdc"
ZED_X_TRANSLATE = (1.0, 0.0, 1.0)
# PDF의 Orient (0, -50, 180). 회전 op 종류가 중요하다 - 아래 '회전 규약' 주석 참고.
ZED_X_ROTATE_ZYX = (0.0, -50.0, 180.0)  # degrees, USD xformOp:rotateZYX

# 캡처 해상도. 실물 ZED-X 는 눈당 1920x1200 이지만, 검출/학습에는 720p로 충분하고
# 렌더 비용이 절반 이하라 시뮬레이션 실시간성 확보에 유리하다.
DEFAULT_RESOLUTION = (1280, 720)


class ZedXCamera:
    """ZED-X 에셋 스폰 + RGB/Depth 취득.

    사용 순서:
        zed = ZedXCamera()
        zed.spawn()          # setup_scene 단계 (프림 생성)
        ...world.reset()...
        zed.initialize()     # reset 이후 (render product + annotator 생성)
        rgb, depth = zed.capture()
    """

    def __init__(
        self,
        prim_path: str = "/World/ZED_X",
        translate=ZED_X_TRANSLATE,
        rotate_zyx=ZED_X_ROTATE_ZYX,
        resolution=DEFAULT_RESOLUTION,
        use_right_eye: bool = False,
    ):
        self.prim_path = prim_path
        self.translate = tuple(float(v) for v in translate)
        self.rotate_zyx = tuple(float(v) for v in rotate_zyx)
        self.resolution = tuple(resolution)
        self.use_right_eye = use_right_eye

        self.camera = None            # isaacsim.sensors.camera.Camera
        self.camera_prim_path = None
        self.info = {}                # 진단용 메타데이터

    # ------------------------------------------------------------------ spawn
    def spawn(self):
        """USD 에셋을 스테이지에 올리고 명세대로 배치한다. (물리 시작 전 호출)"""
        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("Isaac 에셋 루트를 찾을 수 없습니다 (네트워크/설정 확인).")
        asset_path = assets_root + ZED_X_USD_SUBPATH
        self.info["asset_path"] = asset_path

        add_reference_to_stage(usd_path=asset_path, prim_path=self.prim_path, prim_type="Xform")
        self._apply_pose()
        self.info["kinematic"] = self._enable_kinematic()

        cameras = self._find_camera_prims()
        self.info["camera_prims"] = cameras
        if not cameras:
            raise RuntimeError(f"{self.prim_path} 안에서 Camera 프림을 찾지 못했습니다.")

        wanted = "right" if self.use_right_eye else "left"
        match = [p for p in cameras if wanted in p.lower()]
        self.camera_prim_path = match[0] if match else cameras[0]
        self.info["camera_prim_path"] = self.camera_prim_path
        return self.camera_prim_path

    def _apply_pose(self):
        """GUI Transform 패널이 하는 것과 동일하게 에셋의 기존 xformOp 값을 덮어쓴다.

        쿼터니언으로 변환하지 않고 에셋이 원래 쓰는 op(rotateZYX)를 그대로 쓰는 이유:
        오일러 순서/내외재 규약 차이로 자세가 통째로 뒤집히는 것을 막기 위함
        (모듈 docstring의 '회전 규약' 실측 결과 참고).
        """
        prim = get_prim_at_path(self.prim_path)
        xform = UsdGeom.Xformable(prim)
        precision = UsdGeom.XformOp.PrecisionDouble  # 에셋이 double3 로 저작돼 있음

        if prim.HasAttribute("xformOp:translate"):
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*self.translate))
        else:
            xform.AddTranslateOp(precision).Set(Gf.Vec3d(*self.translate))

        if prim.HasAttribute("xformOp:rotateZYX"):
            prim.GetAttribute("xformOp:rotateZYX").Set(Gf.Vec3d(*self.rotate_zyx))
        else:
            xform.AddRotateZYXOp(precision).Set(Gf.Vec3d(*self.rotate_zyx))

        self.info["xform_op_order"] = list(xform.GetXformOpOrderAttr().Get() or [])

    def _enable_kinematic(self) -> str:
        """명세의 'Kinematic Enabled: Check' 재현.

        ZED_X 루트 프림에는 이미 PhysicsRigidBodyAPI 가 적용돼 있다. 키네마틱을 켜면
        중력/충돌에 반응하지 않고 고정된다 - 고정 설치 카메라이므로 필수.
        (끄면 카메라가 바닥으로 떨어져 시점이 완전히 달라진다)
        """
        root = get_prim_at_path(self.prim_path)
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
                return f"kinematicEnabled=True @ {prim.GetPath()}"
        UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
        return f"RigidBodyAPI 신규 적용 + kinematicEnabled=True @ {root.GetPath()}"

    def _find_camera_prims(self):
        root = get_prim_at_path(self.prim_path)
        return [str(p.GetPath()) for p in Usd.PrimRange(root) if str(p.GetTypeName()) == "Camera"]

    # ------------------------------------------------------------- initialize
    def initialize(self):
        """render product 생성 + annotator 부착. (world.reset() 이후 호출)"""
        self.camera = Camera(prim_path=self.camera_prim_path, resolution=self.resolution)
        self.camera.initialize()  # rgb annotator 자동 부착

        # 깊이: distance_to_image_plane = 이미지 평면까지의 '수직' 거리(Z_cam).
        # Isaac Sim 역투영이 K^-1 @ [u*d, v*d, d] 를 쓰므로 이 값이 정확히 맞는 입력이다.
        # (distance_to_camera 는 광선 거리라서 쓰면 안 된다)
        self.camera.attach_annotator("distance_to_image_plane")

        # 역투영 API 3종은 pinhole 을 요구한다. ZED_X 카메라 프림에는 왜곡 스키마가 없어
        # get_lens_distortion_model() 이 기본값 "pinhole" 을 반환한다 -> 추가 조치 불필요.
        self.info["lens_model"] = self.camera.get_lens_distortion_model()
        self.info["resolution"] = tuple(self.camera.get_resolution())
        return self.camera

    # ---------------------------------------------------------------- capture
    def capture(self):
        """(rgb uint8 (H,W,3), depth float32 (H,W)) 반환. 아직 준비 안 됐으면 (None, None)."""
        rgb = self.camera.get_rgb()
        depth = self.camera.get_depth()
        if rgb is None or depth is None:
            return None, None
        rgb = np.asarray(rgb)
        depth = np.asarray(depth)
        if rgb.size == 0 or depth.size == 0:
            return None, None
        return rgb[:, :, :3].astype(np.uint8), depth.astype(np.float32)

    # ------------------------------------------------------------------ utils
    @property
    def position(self):
        """카메라 광학 중심의 월드 좌표 (역투영 보정에서 광선의 원점)."""
        return np.asarray(self.camera.get_world_pose()[0], dtype=np.float64)

    def describe(self) -> str:
        lines = ["[ZED-X]"]
        for k, v in self.info.items():
            lines.append(f"    {k:20s}: {v}")
        if self.camera is not None:
            pos, quat = self.camera.get_world_pose()
            lines.append(f"    {'world pos':20s}: {np.round(np.asarray(pos), 4)}")
            lines.append(f"    {'world quat(wxyz)':20s}: {np.round(np.asarray(quat), 4)}")
            try:
                K = np.asarray(self.camera.get_intrinsics_matrix())
                lines.append(f"    {'fx, fy':20s}: {K[0,0]:.2f}, {K[1,1]:.2f}")
                lines.append(f"    {'cx, cy':20s}: {K[0,2]:.2f}, {K[1,2]:.2f}")
                lines.append(f"    {'FOV h,v (deg)':20s}: {np.degrees(self.camera.get_horizontal_fov()):.1f}, "
                             f"{np.degrees(self.camera.get_vertical_fov()):.1f}")
            except Exception as exc:  # pragma: no cover
                lines.append(f"    intrinsics 조회 실패: {exc}")
        return "\n".join(lines)
