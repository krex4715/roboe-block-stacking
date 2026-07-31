# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# [ROBOE Take-home] isaacsim.examples.interactive.franka_cortex.franka_cortex 의 수정본.
# 원본 대비 변경점은 "[ROBOE]" 주석으로 표시한다.
#   - M0: 클래스명 변경 외 원본과 동일 (스톡 동작 검증용 베이스라인)
#   - M2: 씬 구성을 scene_setup 으로 공통화(연두 큐브 + 조명 + 타워 표식),
#         ZED-X 카메라 스폰/초기화, tower_position 을 UI에서 받아 behavior 에 전달
#   - M4/M5 예정: YOLO 추론 + 3D 추정 + Cortex bridge 연결

import gc

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
from .perception.zed_camera import ZedXCamera
from .scene_setup import add_cubes, add_lighting, add_tower_marker

DEFAULT_TOWER_POSITION = np.array([0.25, 0.30, 0.0])


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

    def __init__(self, monitor_fn=None):
        super().__init__()
        self._monitor_fn = monitor_fn
        self.behavior = None
        self.robot = None
        self.decider_network = None
        self.context_monitor = ContextStateMonitor(print_dt=0.25, diagnostic_fn=self._on_monitor_update)
        # [ROBOE] UI에서 덮어쓴다. 과제의 "적재 목표 위치는 사용자가 임의로 설정"에 대응.
        self.tower_position = np.array(DEFAULT_TOWER_POSITION)
        self.zed = None

    def setup_scene(self):
        world = self.get_world()
        self.robot = world.add_robot(add_franka_to_stage(name="franka", prim_path="/World/Franka"))

        # [ROBOE] 큐브 스펙/조명을 scene_setup 으로 공통화.
        # SDG(학습 이미지)와 런타임(추론 이미지)이 같은 코드로 만들어져야 도메인 갭이 없다.
        cubes = add_cubes(world.scene)
        for obj in cubes.values():
            # register_obstacle 이 이 큐브들을 behavior 가 보는 world model 에 넣는다
            # (BuildTowerContext.reset() 이 robot.registered_obstacles 를 CortexObject 로 감쌈)
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

        await omni.kit.app.get_app().next_update_async()

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
        gc.collect()
        await super().load_world_async()

    def world_cleanup(self):
        """월드가 정리될 때 우리 쪽 참조도 함께 버린다 (죽은 프림 참조 방지)."""
        self.robot = None
        self.zed = None
        self.decider_network = None
