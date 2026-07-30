# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# [ROBOE Take-home] isaacsim.examples.interactive.franka_cortex.franka_cortex 의 수정본.
# 원본 대비 변경점은 "[ROBOE]" 주석으로 표시한다.
#   - M0: 클래스명 변경 외 원본과 동일 (스톡 동작 검증용 베이스라인)
#   - M1 예정: ZED-X 카메라 스폰 + RGB-D 취득
#   - M2 예정: 연두색 큐브, 타워 위치 파라미터화
#   - M4/M5 예정: YOLO 추론 + 3D 추정 + Cortex bridge 연결

import carb
import numpy as np
import omni
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.cortex.framework.cortex_utils import load_behavior_module
from isaacsim.cortex.framework.cortex_world import Behavior, CortexWorld, LogicalStateMonitor
from isaacsim.cortex.framework.dfb import DfDiagnosticsMonitor
from isaacsim.cortex.framework.robot import CortexFranka, add_franka_to_stage
from isaacsim.cortex.framework.tools import SteadyRate
from isaacsim.examples.interactive.cortex.cortex_base import CortexBase


class CubeSpec:
    def __init__(self, name, color):
        self.name = name
        self.color = np.array(color)


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
        self.context_monitor = ContextStateMonitor(print_dt=0.25, diagnostic_fn=self._on_monitor_update)

    def setup_scene(self):
        world = self.get_world()
        self.robot = world.add_robot(add_franka_to_stage(name="franka", prim_path="/World/Franka"))

        # 큐브 4개: 이름이 behavior의 desired_stack(["...Cube"])과 일치해야 한다.
        obs_specs = [
            CubeSpec("RedCube", [0.7, 0.0, 0.0]),
            CubeSpec("BlueCube", [0.0, 0.0, 0.7]),
            CubeSpec("YellowCube", [0.7, 0.7, 0.0]),
            CubeSpec("GreenCube", [0.0, 0.7, 0.0]),
        ]
        width = 0.0515
        for i, (x, spec) in enumerate(zip(np.linspace(0.3, 0.7, len(obs_specs)), obs_specs)):
            obj = world.scene.add(
                DynamicCuboid(
                    prim_path="/World/Obs/{}".format(spec.name),
                    name=spec.name,
                    size=width,
                    color=spec.color,
                    position=np.array([x, -0.4, width / 2]),
                )
            )
            # register_obstacle이 이 큐브들을 behavior가 보는 world model에 넣는다
            # (BuildTowerContext.reset()이 robot.registered_obstacles를 CortexObject로 감쌈)
            self.robot.register_obstacle(obj)
        world.scene.add_default_ground_plane()

    async def load_behavior(self, behavior):
        world = self.get_world()
        self.behavior = behavior
        self.decider_network = load_behavior_module(self.behavior).make_decider_network(self.robot)
        self.decider_network.context.add_monitor(self.context_monitor.monitor)
        world.add_decider_network(self.decider_network)

    def clear_behavior(self):
        world = self.get_world()
        world._logical_state_monitors.clear()
        world._behaviors.clear()

    async def setup_post_load(self, soft=False):
        world = self.get_world()
        prim_path = "/World/Franka"
        if not self.robot:
            self.robot = world._robots["franka"]
        self.decider_network = load_behavior_module(self.behavior).make_decider_network(self.robot)
        self.decider_network.context.add_monitor(self.context_monitor.monitor)
        world.add_decider_network(self.decider_network)
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

    def world_cleanup(self):
        pass
