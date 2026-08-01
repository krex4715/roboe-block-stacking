# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# [ROBOE Take-home] isaacsim.examples.interactive.franka_cortex.franka_cortex 의 수정본.
# 원본 대비 변경점은 "[ROBOE]" 주석으로 표시한다.
#   - M0: 클래스명 변경 외 원본과 동일 (스톡 동작 검증용 베이스라인)
#   - M2: 씬 구성을 scene_setup 으로 공통화(연두 큐브 + 조명 + 타워 표식),
#         ZED-X 카메라 스폰/초기화, tower_position 을 UI에서 받아 behavior 에 전달
#   - M4: YOLO 추론 + 3D 추정을 GUI 루프에 연결 (뷰포트에 인식 결과를 3D 마커로 표시)
#   - M5 예정: Cortex bridge 연결 (set_measured_pose)

import gc
import json
from pathlib import Path

import carb
import numpy as np
import omni
from isaacsim.cortex.framework.cortex_utils import load_behavior_module
from isaacsim.cortex.framework.cortex_world import Behavior, CortexWorld, LogicalStateMonitor
from isaacsim.cortex.framework.dfb import DfDiagnosticsMonitor
from isaacsim.cortex.framework.robot import CortexFranka, add_franka_to_stage
from isaacsim.cortex.framework.tools import SteadyRate
from isaacsim.examples.interactive.cortex.cortex_base import CortexBase





# [ROBOE] 씬/비전 공통 모듈. 상대 import 라 GUI 확장 경로와 standalone 양쪽에서 동작한다.
from .perception.cortex_bridge import CortexPerceptionBridge
from .perception.estimator_3d import (
    backproject_pixels,
    clamp_to_support,
    estimate_yaw,
    sample_depth,
    surface_to_center,
)
from .perception.zed_camera import ZedXCamera
from .scene_setup import add_belief_cubes, add_cubes, add_lighting, add_tower_marker

DEFAULT_TOWER_POSITION = np.array([0.25, 0.30, 0.0])

# 저장소 루트 (심링크를 풀어 실제 경로 기준으로 계산)
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "best.torchscript"
MODEL_META = REPO_ROOT / "models" / "model_meta.json"

# 물리 60Hz 기준. 6스텝마다 = 10Hz 로 인식을 돌린다.
# 근거: Isaac Sim 렌더와 GPU 를 공유하면 추론이 ~27ms 걸린다(단독 2.2ms). 10Hz면 예산의 27%.
PERCEPTION_EVERY_N_STEPS = 6

# 뷰포트 마커 색 (인식된 클래스별)
MARKER_COLORS = {
    "red_cube": (1.0, 0.2, 0.2, 1.0),
    "yellow_cube": (1.0, 1.0, 0.2, 1.0),
    "green_cube": (0.5, 1.0, 0.2, 1.0),
    "blue_cube": (0.3, 0.4, 1.0, 1.0),
}
BLOCK_TO_CLASS = {"RedCube": "red_cube", "YellowCube": "yellow_cube",
                  "GreenCube": "green_cube", "BlueCube": "blue_cube"}


class ContextStateMonitor(DfDiagnosticsMonitor):
    """Behavior context의 diagnostics_message를 주기적으로 읽어 UI로 전달하는 모니터."""

    def __init__(self, print_dt, diagnostic_fn=None):
        super().__init__(print_dt=print_dt)
        self.diagnostic_fn = diagnostic_fn

    def print_diagnostics(self, context):
        if self.diagnostic_fn:
            self.diagnostic_fn(context)


class RoboeBlockStacking(CortexBase):
    """[ROBOE] FrankaCortex 예제의 수정본. M0 단계에서는 스톡과 동일하게 동작해야 한다."""

    def __init__(self, monitor_fn=None, perception_fn=None):
        super().__init__()
        self._monitor_fn = monitor_fn
        self._perception_fn = perception_fn  # [ROBOE] 인식 결과 텍스트를 UI 로 보내는 콜백
        self.behavior = None
        self.robot = None
        self.decider_network = None
        self.context_monitor = ContextStateMonitor(print_dt=0.25, diagnostic_fn=self._on_monitor_update)
        # [ROBOE] UI에서 덮어쓴다. 과제의 "적재 목표 위치는 사용자가 임의로 설정"에 대응.
        self.tower_position = np.array(DEFAULT_TOWER_POSITION)
        self.zed = None

        # [ROBOE] 인식 파이프라인 상태
        self.detector = None
        self.perception_enabled = True
        self.last_estimates = {}       # {클래스: 추정 3D 위치}
        self._perception_step = 0
        self._debug_draw = None
        # [ROBOE] 깊이->중심 보정 방식. UI에서 바꾸면 뷰포트의 점이 실제로 움직인다.
        # "none" 으로 두면 점이 큐브보다 ~3cm 카메라 쪽으로 앞서 찍히는 것을 눈으로 볼 수 있다.
        self.correction_mode = "box"

        # [ROBOE] belief 구조 (scene_setup.add_belief_cubes 주석 참고)
        #   "ghost"  : 로봇은 고스트(=인식 결과)만 보고 계획, 물리적으로는 진짜 큐브를 집는다
        #              -> 인식 오차가 곧 파지 오차. 인식이 진짜로 루프 안에 있음을 증명 가능
        #   "direct" : 물리 큐브를 그대로 등록 (스톡 동작 = 인식 없이도 완벽히 동작하는 비교군)
        self.belief_mode = "ghost"
        self.cubes = {}          # 물리 큐브
        self.belief_cubes = {}   # 고스트
        self.bridge = CortexPerceptionBridge()

    def setup_scene(self):
        world = self.get_world()
        self.robot = world.add_robot(add_franka_to_stage(name="franka", prim_path="/World/Franka"))

        # [ROBOE] 큐브 스펙/조명을 scene_setup 으로 공통화.
        # SDG(학습 이미지)와 런타임(추론 이미지)이 같은 코드로 만들어져야 도메인 갭이 없다.
        #
        # register_obstacle 이 "behavior 가 보는 세계"를 정한다
        # (BuildTowerContext.reset() 이 robot.registered_obstacles 를 CortexObject 로 감쌈).
        # 무엇을 등록하느냐가 곧 belief 구조 선택이다.
        if self.belief_mode == "ghost":
            self.cubes = add_cubes(world.scene, name_suffix="Real")  # 카메라가 보는 진짜 큐브
            self.belief_cubes = add_belief_cubes(world.scene)        # 로봇이 믿는 고스트
            for obj in self.belief_cubes.values():
                self.robot.register_obstacle(obj)
        else:
            self.cubes = add_cubes(world.scene)
            self.belief_cubes = {}
            for obj in self.cubes.values():
                self.robot.register_obstacle(obj)

        world.scene.add_default_ground_plane()
        add_lighting()
        # 적재 목표 위치 표식(시각 전용). obstacle 로 등록하지 않는다 - 등록하면 behavior 가
        # 이것도 쌓을 블록으로 오인한다.
        add_tower_marker(self.tower_position, world.scene)

        # [ROBOE] ZED-X 를 코드로 배치한다. 손으로 놓으면 LOAD 할 때마다 사라지고
        # 뷰포트 조작으로 자세가 틀어질 수 있어 재현성이 없다.
        self.zed = ZedXCamera()
        self.zed.spawn()

    async def load_behavior(self, behavior):
        world = self.get_world()
        self.behavior = behavior
        self.decider_network = load_behavior_module(self.behavior).make_decider_network(
            self.robot, tower_position=self.tower_position
        )
        self.decider_network.context.add_monitor(self.context_monitor.monitor)
        world.add_decider_network(self.decider_network)

    def clear_behavior(self):
        world = self.get_world()
        world._logical_state_monitors.clear()
        world._behaviors.clear()

    async def setup_post_load(self, soft=False):
        world = self.get_world()
        if not self.robot:
            self.robot = world._robots["franka"]
        self.decider_network = load_behavior_module(self.behavior).make_decider_network(
            self.robot, tower_position=self.tower_position
        )
        self.decider_network.context.add_monitor(self.context_monitor.monitor)
        world.add_decider_network(self.decider_network)

        # [ROBOE] 카메라 initialize 는 반드시 world.reset() **이후**여야 한다
        # (render product 를 스테이지에 붙이는 작업이라서). base_sample 의 로드 순서:
        #   setup_scene() -> world.reset_async() -> setup_post_load()  <- 여기
        if self.zed is not None:
            self.zed.initialize()
            carb.log_info(f"[ROBOE] ZED-X ready: {self.zed.info}")
            self._load_detector()

        await omni.kit.app.get_app().next_update_async()

    # ------------------------------------------------------- [ROBOE] 인식 루프
    def _load_detector(self):
        """TorchScript 검출기를 로드한다. 모델이 없으면 인식 없이 GT 로 동작한다."""
        if self.detector is not None:
            return
        if not MODEL_PATH.exists():
            self._report_perception(f"모델 없음: {MODEL_PATH}\n"
                                    "-> training/train.py 를 먼저 실행하세요. "
                                    "(지금은 인식 없이 ground truth 로 동작합니다)")
            return
        try:
            from .perception.detector import CubeDetector

            meta = json.loads(MODEL_META.read_text()) if MODEL_META.exists() else {}
            names = meta.get("class_names", ["red_cube", "yellow_cube", "green_cube", "blue_cube"])
            self.detector = CubeDetector(str(MODEL_PATH), names,
                                         imgsz=int(meta.get("imgsz", 640)), conf=0.5)
            m50 = meta.get("mAP50")
            self._report_perception(f"검출기 로드 완료: {MODEL_PATH.name}\n"
                                    f"클래스: {', '.join(names)}\n"
                                    f"val mAP50 = {m50:.4f}\n\n"
                                    "Start 를 누르면 인식이 시작됩니다."
                                    if isinstance(m50, float) else
                                    f"검출기 로드 완료: {MODEL_PATH.name}\n\nStart 를 누르세요.")
        except Exception as exc:
            carb.log_warn(f"[ROBOE] 검출기 로드 실패: {exc}")
            self._report_perception(f"검출기 로드 실패: {exc}")

    def _get_debug_draw(self):
        if self._debug_draw is None:
            try:
                from isaacsim.util.debug_draw import _debug_draw

                self._debug_draw = _debug_draw.acquire_debug_draw_interface()
            except Exception as exc:
                carb.log_warn(f"[ROBOE] debug_draw 사용 불가: {exc}")
                self._debug_draw = False
        return self._debug_draw or None

    def _run_perception(self):
        """RGB-D 취득 -> YOLO 검출 -> 3D 위치 추정 -> 뷰포트 마커 + UI 텍스트."""
        if self.detector is None or self.zed is None or self.zed.camera is None:
            return
        rgb, depth = self.zed.capture()
        if rgb is None or depth is None:
            return

        detections = self.detector(rgb)
        best = self.detector.best_per_class(detections)

        cam_pos = self.zed.position
        estimates, yaws, lines = {}, {}, []
        for cls, det in sorted(best.items()):
            x0, y0, x1, y1 = det["box"]
            u, v = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            depth_val = sample_depth(depth, (u, v), box=(x0, y0, x1, y1))
            if depth_val is None:
                lines.append(f"{cls:12s} score {det['score']:.2f}  (깊이 없음)")
                continue
            p_surf = backproject_pixels(self.zed.camera, [[u, v]], [depth_val])[0]
            p = surface_to_center([p_surf], cam_pos, mode=self.correction_mode)[0]
            if self.correction_mode != "none":
                p = clamp_to_support([p])[0]
            estimates[cls] = p

            # yaw 추정 - 없으면 belief 방향 유지 (배치 평가에서 45도 큐브 파지가
            # 모서리를 쳐내는 실패로 필요성이 입증됨. 과제의 "필요 시 Orientation")
            yaw, _n = estimate_yaw(self.zed.camera, depth, (x0, y0, x1, y1))
            if yaw is not None:
                yaws[cls] = yaw
            yaw_s = f" yaw {np.degrees(yaw):4.1f}°" if yaw is not None else ""
            lines.append(f"{cls:12s} {det['score']:.2f}  "
                         f"({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}){yaw_s}")

        self.last_estimates = estimates
        self._draw_estimates(estimates)

        # [ROBOE] 여기가 인식이 실제 제어에 연결되는 지점.
        # bridge 가 CortexObject.set_measured_pose() 로 발행하면
        # behavior 의 monitor_perception() 이 belief 를 동기화한다.
        bridge_line = ""
        ctx = getattr(self.decider_network, "context", None) if self.decider_network else None
        if ctx is not None:
            # direct 모드에서는 belief == 물리 큐브라 직접 쓰면 큐브가 순간이동한다.
            # 측정값만 발행하고 동기화는 프레임워크(monitor_perception)에 맡긴다.
            self.bridge.direct_write = (self.belief_mode == "ghost")
            self.bridge.update(ctx, best, estimates, self.robot, yaws=yaws)
            bridge_line = "\n" + self.bridge.summary()
            for name, reason in sorted(self.bridge.last_reasons.items()):
                if reason != "발행":
                    bridge_line += f"\n  {name:11s} {reason}"

        header = (f"검출 {len(best)}/4   추론 {self.detector.last_latency_ms:.1f} ms   "
                  f"@ {60.0/PERCEPTION_EVERY_N_STEPS:.0f} Hz   보정={self.correction_mode}   "
                  f"belief={self.belief_mode}\n" + "-" * 52)
        self._report_perception(header + "\n" + ("\n".join(lines) if lines else "(검출 없음)")
                                + bridge_line)

    def _draw_estimates(self, estimates):
        """뷰포트에 두 가지를 그린다 - 이 둘의 차이가 곧 파이프라인의 상태를 보여준다.

          작은 점 : 이번 프레임의 **원시 인식 결과**
          큰 점   : 로봇이 **믿고 있는 위치**(고스트) — 게이팅/필터를 거친 결과

        평소엔 겹쳐 보이지만, 카메라를 가리거나 게이트가 걸리면 큰 점만 남는다.
        즉 "로봇이 무엇을 보고 계획하는가"가 눈에 보인다. (고스트 프림 자체는
        카메라가 자기 자신을 검출하지 않도록 렌더링을 꺼두었다)
        """
        draw = self._get_debug_draw()
        if draw is None:
            return
        draw.clear_points()
        pts, colors, sizes = [], [], []

        for cls, p in estimates.items():
            pts.append((float(p[0]), float(p[1]), float(p[2]) + 0.05))
            colors.append(MARKER_COLORS.get(cls, (1.0, 1.0, 1.0, 1.0)))
            sizes.append(8)

        for name, ghost in self.belief_cubes.items():
            cls = BLOCK_TO_CLASS.get(name)
            p, _ = ghost.get_world_pose()
            pts.append((float(p[0]), float(p[1]), float(p[2]) + 0.09))
            colors.append(MARKER_COLORS.get(cls, (1.0, 1.0, 1.0, 1.0)))
            sizes.append(20)

        if pts:
            draw.draw_points(pts, colors, sizes)

    def _report_perception(self, text):
        if self._perception_fn:
            self._perception_fn(text)

    def set_belief_mode(self, mode: str):
        """belief 구조 전환. 씬 구성이 달라지므로 다음 LOAD 부터 적용된다."""
        if mode in ("ghost", "direct"):
            self.belief_mode = mode

    def set_correction_mode(self, mode: str):
        """깊이->큐브중심 보정 방식 전환 (box / ray / none).

        발표 데모용: none 으로 바꾸면 뷰포트의 점이 큐브에서 카메라 쪽으로 ~3cm 튀어나온다.
        "깊이는 앞면을 준다"를 말이 아니라 화면으로 보여줄 수 있다.
        """
        if mode in ("box", "ray", "none"):
            self.correction_mode = mode

    def set_perception_enabled(self, enabled: bool):
        self.perception_enabled = bool(enabled)
        if not enabled:
            draw = self._get_debug_draw()
            if draw is not None:
                draw.clear_points()
            self._report_perception("인식 OFF — 로봇은 시뮬레이터 ground truth 로 동작합니다.")

    def _on_monitor_update(self, context):
        diagnostic = ""
        decision_stack = ""
        if hasattr(context, "diagnostics_message"):
            diagnostic = context.diagnostics_message
        if self.decider_network._decider_state.stack:
            decision_stack = "\n".join(
                [
                    "{0}{1}".format("  " * i, element)
                    for i, element in enumerate(str(i) for i in self.decider_network._decider_state.stack)
                ]
            )

        if self._monitor_fn:
            self._monitor_fn(diagnostic, decision_stack)

    def _on_physics_step(self, step_size):
        world = self.get_world()
        world.step(False, False)

        # [ROBOE] 파지 추적/부착은 **매 스텝** 돌려야 한다.
        # in_gripper 가 매 스텝 깜빡이는 신호라 10Hz 로 샘플링하면 엇박이 나 부착이 안 걸린다.
        if (self.perception_enabled and self.belief_mode == "ghost"
                and self.decider_network is not None):
            try:
                self.bridge.tick(self.decider_network.context, self.robot)
            except Exception as exc:
                carb.log_warn(f"[ROBOE] bridge.tick 실패: {exc}")

        # [ROBOE] 인식(무거움)은 주기적으로만 (위 PERCEPTION_EVERY_N_STEPS 주석 참고)
        self._perception_step += 1
        if self.perception_enabled and self._perception_step % PERCEPTION_EVERY_N_STEPS == 0:
            try:
                self._run_perception()
            except Exception as exc:
                carb.log_warn(f"[ROBOE] 인식 실패: {exc}")
                self._report_perception(f"인식 오류: {exc}")

    async def on_event_async(self):
        world = self.get_world()
        await omni.kit.app.get_app().next_update_async()
        world.reset_cortex()
        world.add_physics_callback("sim_step", self._on_physics_step)
        await world.play_async()

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")

    # ------------------------------------------------------------------ [ROBOE]
    async def load_world_async(self):
        """LOAD 를 누를 때마다 씬을 확실히 새로 만든다.

        왜 필요한가 (베이스 예제의 잠재 버그):
        CortexBase.load_world_async 는 `if CortexWorld.instance() is None:` 일 때만
        setup_scene() 을 부른다. 그런데 월드 싱글톤은 SimulationContext._instance 하나를
        모든 하위 클래스가 공유하므로, **다른 예제를 로드했다가 돌아오면** 인스턴스가 살아남아
        setup_scene() 이 통째로 건너뛰어진다. 반면 스테이지는 create_new_stage_async() 로
        이미 비워진 상태라, robot / 큐브 / 카메라 참조가 전부 '사라진 프림'을 가리키게 된다.
        그 상태로 setup_post_load() 가 돌면 첫 접근에서 예외가 나는데, async 태스크 안이라
        GUI 콘솔에는 "Task exception was never retrieved" 로만 보여 원인 파악이 어렵다.

        그래서 이전 월드를 정리하고 참조를 비운 뒤 부모 구현에 넘긴다.
        (스톡 Franka Cortex Examples 도 같은 구조라 동일한 증상이 난다)
        """
        if self._world is not None:
            self._world_cleanup()
            self._world.clear_instance()
            self._world = None
        elif CortexWorld.instance() is not None:
            CortexWorld.instance().clear_instance()

        self.robot = None
        self.zed = None
        self.decider_network = None
        self.detector = None
        self.cubes = {}
        self.belief_cubes = {}
        self.bridge.reset()
        gc.collect()
        await super().load_world_async()

    def world_cleanup(self):
        """월드가 정리될 때 우리 쪽 참조도 함께 버린다 (죽은 프림 참조 방지)."""
        self.robot = None
        self.zed = None
        self.decider_network = None
