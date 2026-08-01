# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# [ROBOE Take-home] isaacsim.examples.interactive.franka_cortex.franka_cortex_extension 의 수정본.
# 원본 대비 변경점:
#   - 예제 이름/카테고리: "ROBOE Block Stacking" / "Custom"
#   - behavior 드롭다운 제거 → 이 패키지의 behavior/block_stacking_behavior.py 복사본을 항상 사용
#     (스톡은 isaacsim.cortex.behaviors 확장 경로에서 로드; 우리는 수정 가능한 로컬 복사본을 로드)

import asyncio
import os

import numpy as np
import omni
import omni.ext
import omni.ui as ui
from isaacsim.cortex.framework.cortex_world import CortexWorld
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from isaacsim.examples.interactive.user_examples.roboe_block_stacking.roboe_stacking_example import (
    DEFAULT_TOWER_POSITION,
    RoboeBlockStacking,
)
from isaacsim.examples.interactive.user_examples.roboe_block_stacking.scene_setup import validate_tower_position
from isaacsim.gui.components.ui_utils import (
    btn_builder,
    cb_builder,
    dropdown_builder,
    float_builder,
    get_style,
    str_builder,
)


class RoboeBlockStackingExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "ROBOE Block Stacking"
        self.category = "Custom"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "ROBOE Block Stacking (AI Vision)",
            "doc_link": "",
            "overview": (
                "[ROBOE Take-home] ZED-X RGB-D + AI 인식 기반 Franka 블록 스태킹.\n"
                "빨강 -> 노랑 -> 연두 -> 파랑 순서로 사용자 지정 위치에 쌓는다.\n"
                "베이스: Franka Cortex Examples (Block Stacking behavior)."
            ),
        }

        ui_handle = RoboeBlockStackingUI(**ui_kwargs)
        ui_handle.sample = RoboeBlockStacking(ui_handle.on_diagnostics, ui_handle.on_perception)
        # 검출 뷰 창은 GUI 에서만 의미가 있으므로 확장(=GUI 전용 코드)에서 켠다.
        # headless 러너들은 RoboeBlockStacking 을 직접 만들므로 기본값 False 그대로다.
        ui_handle.sample.enable_detection_view = True

        get_browser_instance().register_example(
            name=self.example_name,
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

    def on_shutdown(self):
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)


class RoboeBlockStackingUI(BaseSampleUITemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # [ROBOE] 스톡과 달리 로컬 behavior 복사본을 고정 사용 (여기를 수정하며 개발)
        self.behavior_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "behavior", "block_stacking_behavior.py"
        )
        self.loaded = False

    def build_ui(self):
        self.task_ui_elements = {}
        self.build_default_frame()

        with self._controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # [ROBOE] 적재 목표 위치 (과제: "사용자가 임의로 적절한 위치를 설정")
                # LOAD 를 누를 때 읽어서 behavior 로 전달한다.
                self.task_ui_elements["Tower X"] = float_builder(
                    "Tower X (m)", default_val=float(DEFAULT_TOWER_POSITION[0]),
                    tooltip="적재 목표 위치 X. 로봇 베이스에서 0.30~0.75m, 큐브 스폰과 0.15m 초과",
                )
                self.task_ui_elements["Tower Y"] = float_builder(
                    "Tower Y (m)", default_val=float(DEFAULT_TOWER_POSITION[1]),
                    tooltip="적재 목표 위치 Y",
                )
                # [ROBOE] 인식 On/Off. 끄면 로봇이 시뮬레이터 ground truth 로 동작하므로
                # "인식이 실제로 일하고 있는가"를 눈으로 비교할 수 있다.
                cb_builder(
                    label="Perception (AI)", default_val=True,
                    tooltip="ZED-X + YOLO 인식 실행. 끄면 ground truth 사용",
                    on_clicked_fn=self._on_perception_toggled,
                )
                # [ROBOE] 깊이->큐브중심 보정 방식. none 으로 바꾸면 뷰포트의 점이
                # 큐브에서 카메라 쪽으로 ~3cm 튀어나오는 것이 눈으로 보인다 (발표 데모용).
                self.task_ui_elements["Correction"] = dropdown_builder(
                    "Depth 보정",
                    items=["box (광선-박스 정확해)", "ray (반큐브)", "none (보정 없음)"],
                    on_clicked_fn=self._on_correction_changed,
                )
                # [ROBOE] belief 구조. LOAD 를 다시 눌러야 적용된다(씬 구성이 달라짐).
                self.task_ui_elements["Belief"] = dropdown_builder(
                    "Belief 구조",
                    items=["ghost (인식이 제어에 반영)", "direct (ground truth 비교군)"],
                    on_clicked_fn=self._on_belief_changed,
                )
                dict = {
                    "label": "Load World",
                    "type": "button",
                    "text": "Load",
                    "tooltip": "Load World and Task",
                    "on_clicked_fn": self._on_load_world,
                }
                self._buttons["Load World"] = btn_builder(**dict)
                self._buttons["Load World"].enabled = True
                dict = {
                    "label": "Reset",
                    "type": "button",
                    "text": "Reset",
                    "tooltip": "Reset robot and environment",
                    "on_clicked_fn": self._on_reset,
                }
                self._buttons["Reset"] = btn_builder(**dict)
                self._buttons["Reset"].enabled = False
                # [ROBOE] 배치 평가의 랜덤 스폰을 라이브 데모로 - 실행 중 눌러도 안전하다
                # (파지 중/탑에 쌓인 큐브는 제외되고, belief 는 인식이 스스로 따라잡는다).
                dict = {
                    "label": "Randomize Cubes",
                    "type": "button",
                    "text": "Randomize",
                    "tooltip": "탑에 없는 큐브를 유효 작업공간(r 0.40~0.75) 안에서 무작위 재배치. "
                               "belief 는 건드리지 않으므로 인식이 재검출로 따라잡는 과정이 보인다",
                    "on_clicked_fn": self._on_randomize,
                }
                self._buttons["Randomize Cubes"] = btn_builder(**dict)
                self._buttons["Randomize Cubes"].enabled = True

        self.build_extra_frames()

    def build_extra_frames(self):
        extra_stacks = self.get_extra_frames_handle()

        with extra_stacks:
            with ui.CollapsableFrame(
                title="Task Control",
                width=ui.Fraction(0.33),
                height=0,
                visible=True,
                collapsed=False,
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                self.build_task_controls_ui()

            # [ROBOE] 인식 결과 패널 - 검출 개수/점수/추정 3D 위치/추론 지연을 실시간 표시
            with ui.CollapsableFrame(
                title="Perception (ZED-X + YOLO)",
                width=ui.Fraction(0.33),
                height=0,
                visible=True,
                collapsed=False,
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            ):
                self.build_perception_ui()

            with ui.CollapsableFrame(
                title="Diagnostic",
                width=ui.Fraction(0.33),
                height=0,
                visible=True,
                collapsed=True,
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                self.build_diagnostic_ui()

    def _on_load_world(self):
        self._sample.behavior = self.get_behavior()
        # [ROBOE] UI의 적재 위치를 읽어 검증 후 sample 에 전달.
        # behavior 는 도달 불가 위치면 조용히 GoHome 으로 빠져 "아무것도 안 하는" 것처럼
        # 보이므로, 여기서 미리 걸러 이유를 로그로 알려준다.
        pos = np.array(
            [
                float(self.task_ui_elements["Tower X"].get_value_as_float()),
                float(self.task_ui_elements["Tower Y"].get_value_as_float()),
                0.0,
            ]
        )
        ok, msg = validate_tower_position(pos)
        if ok:
            self._sample.tower_position = pos
            print(f"[ROBOE] 적재 목표 위치: {pos[:2]} - {msg}")
        else:
            print(f"[ROBOE] ⚠ 적재 위치 거부: {msg} -> 기본값 {DEFAULT_TOWER_POSITION[:2]} 사용")
            self._sample.tower_position = np.array(DEFAULT_TOWER_POSITION)
            self.task_ui_elements["Tower X"].set_value(float(DEFAULT_TOWER_POSITION[0]))
            self.task_ui_elements["Tower Y"].set_value(float(DEFAULT_TOWER_POSITION[1]))

        self.loaded = True
        super()._on_load_world()

    def on_diagnostics(self, diagnostic, decision_stack):
        if diagnostic:
            self.diagostic_model.set_value(diagnostic)
        self.state_model.set_value(decision_stack)
        self.diagnostics_panel.visible = bool(diagnostic)

    def get_world(self):
        return CortexWorld.instance()

    def get_behavior(self):
        return self.behavior_path

    def _on_start_button_event(self):
        asyncio.ensure_future(self.sample.on_event_async())
        self.task_ui_elements["Start"].enabled = False

    def post_reset_button_event(self):
        self.task_ui_elements["Start"].enabled = True

    def post_load_button_event(self):
        self.task_ui_elements["Start"].enabled = True

    def post_clear_button_event(self):
        self.task_ui_elements["Start"].enabled = False

    def build_task_controls_ui(self):
        with ui.VStack(spacing=5):
            dict = {
                "label": "Start",
                "type": "button",
                "text": "Start",
                "tooltip": "Start",
                "on_clicked_fn": self._on_start_button_event,
            }
            self.task_ui_elements["Start"] = btn_builder(**dict)
            self.task_ui_elements["Start"].enabled = False

    def build_perception_ui(self):
        with ui.VStack(spacing=5):
            ui.Label("인식 결과 (클래스 / 점수 / 추정 3D 위치)", height=20)
            self.perception_model = ui.SimpleStringModel(
                "LOAD 를 누르면 검출기를 로드합니다.\n"
                "Start 후 뷰포트에 인식 위치가 점으로 표시됩니다."
            )
            ui.StringField(self.perception_model, multiline=True, height=150, read_only=True)

    def on_perception(self, text):
        if hasattr(self, "perception_model"):
            self.perception_model.set_value(text)

    def _on_randomize(self):
        try:
            self._sample.randomize_cubes()
        except Exception as exc:
            import carb

            carb.log_warn(f"[ROBOE] 랜덤화 실패: {exc}")

    def _on_perception_toggled(self, value):
        if self._sample is not None:
            self._sample.set_perception_enabled(bool(value))

    def _on_correction_changed(self, value):
        """dropdown_builder 는 선택된 항목 문자열을 넘겨준다. 앞 토큰이 모드명."""
        if self._sample is not None:
            self._sample.set_correction_mode(str(value).split()[0])

    def _on_belief_changed(self, value):
        if self._sample is not None:
            mode = str(value).split()[0]
            self._sample.set_belief_mode(mode)
            self.on_perception(f"belief 구조를 '{mode}' 로 바꿨습니다.\n"
                               "씬 구성이 달라지므로 **LOAD 를 다시 눌러야** 적용됩니다.")

    def build_diagnostic_ui(self):
        with ui.VStack(spacing=5):
            ui.Label("Decision Stack", height=20)
            self.state_model = ui.SimpleStringModel()
            ui.StringField(self.state_model, multiline=True, height=120)
            self.diagnostics_panel = ui.VStack(spacing=5)
            with self.diagnostics_panel:
                ui.Label("Diagnostic message", height=20)
                self.diagostic_model = ui.SimpleStringModel()
                ui.StringField(self.diagostic_model, multiline=True, height=200)
