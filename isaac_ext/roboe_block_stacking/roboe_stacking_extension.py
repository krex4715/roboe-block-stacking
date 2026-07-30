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

import omni
import omni.ext
import omni.ui as ui
from isaacsim.cortex.framework.cortex_world import CortexWorld
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from isaacsim.examples.interactive.user_examples.roboe_block_stacking.roboe_stacking_example import RoboeBlockStacking
from isaacsim.gui.components.ui_utils import btn_builder, cb_builder, dropdown_builder, get_style, str_builder


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
        ui_handle.sample = RoboeBlockStacking(ui_handle.on_diagnostics)

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

            with ui.CollapsableFrame(
                title="Diagnostic",
                width=ui.Fraction(0.33),
                height=0,
                visible=True,
                collapsed=False,
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                self.build_diagnostic_ui()

    def _on_load_world(self):
        self._sample.behavior = self.get_behavior()
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
