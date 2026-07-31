"""[ROBOE] 인식 결과를 Cortex 의 belief 로 연결하는 다리.

**연결 지점**: `CortexObject.set_measured_pose()`.
Cortex 는 belief(로봇이 믿는 세계) 와 measured(인식 결과) 를 분리하도록 설계돼 있고,
behavior 의 `monitor_perception()` 이 measured -> belief 동기화 로직을 이미 갖고 있다.
다만 기본 설치본에는 이 API 를 호출하는 코드가 **한 곳도 없다**. 이 파일이 그 자리를 채운다.

------------------------------------------------------------------------------
설계 원칙 (여섯 번의 실패 실험에서 도출) — **"조작 중에는 belief 를 건드리지 않는다"**
------------------------------------------------------------------------------
인식은 로봇이 물체를 **만지고 있지 않을 때만** 권위를 갖는다. 만지는 순간부터는
조작 시스템이 belief 의 주인이다. 실제 로봇 시스템의 표준 설계이기도 하고,
Cortex 의 `monitor_perception` 가드(`belief 가 EE 근처면 sync 안 함)도 같은 전제다.

실측으로 확인한 실패 모드:

  (a) 접근 중 belief 를 계속 갱신 -> **로봇이 큐브에 영원히 도달하지 못한다.**
      belief 가 목표를 정의하는데 10Hz 로 다시 쓰면 목표가 계속 미세하게 움직여
      `DfGoTarget` 의 도달 판정이 나지 않는다.
      (실측: `approach_grasp > go_target(False)` 로 100초 이상 정체)
      -> 대책: 데드밴드 + **접근 중인 블록 갱신 동결**

  (b) 운반 중 belief 를 갱신하지 않음 -> **탑이 0층에서 진행되지 않는다.**
      고스트는 물리가 없어 아무도 안 옮겨준다.
      -> 대책: 파지 순간의 상대 자세를 기록해 **그리퍼에 강체 부착**

  (c) 파지 판정을 인식 주기(10Hz)로 샘플링 -> **부착이 한 번도 발동하지 않는다.**
      `context.in_gripper` 는 매 물리 스텝 켜졌다 꺼지는 깜빡이 신호라 10Hz 샘플링과
      엇박이 난다.
      -> 대책: 부착 판정은 **매 물리 스텝**(`tick`), 인식 발행은 10Hz(`update`) 로 분리

  (d) direct 모드(belief == 물리 큐브)에서 belief 를 직접 쓰기 -> **큐브가 순간이동**해
      물리가 교란된다.
      -> 대책: direct 모드에서는 measured 만 발행하고 동기화는 프레임워크에 맡긴다

안전장치:
  1. 신뢰도 게이트 + 클래스별 1개 : 유령/중복 검출
  2. EMA 필터                     : 프레임 간 지터
  3. 데드밴드                     : (a) 수렴 실패 방지
  4. 접근/파지 중 동결 + 부착      : (a)(b)(c)
  5. 탑 보호 + 큰 불일치 예외      : 다 쌓은 탑을 노이즈가 흔드는 것. 단 belief 가 확실히
                                    틀렸다고 인식이 말하면 보호를 풀어 자가 교정
  6. z 클램프                     : 지면 관통
"""

import time
import numpy as np

from .estimator_3d import CUBE_HALF

CLASS_TO_BLOCK = {
    "red_cube": "RedCube",
    "yellow_cube": "YellowCube",
    "green_cube": "GreenCube",
    "blue_cube": "BlueCube",
}


class CortexPerceptionBridge:
    """인식 추정치를 Cortex belief 로 반영한다 (조작 중에는 물러난다)."""

    # 작업공간 사전지식: 큐브는 테이블 위 팔 도달범위 안에만 물리적으로 존재할 수 있다.
    # 이 밖을 가리키는 추정은 추정이 아니라 오류다 (가림 중 오검출 + 배경 깊이의 조합).
    # 실측 사례: 배치 직후 팔이 위에 있는 순간 유령 검출이 배경 깊이(3.5m)를 물어
    # est=(-1.38, 0.45) 를 내놓았고, 탑보호의 '큰 불일치 교정' 예외가 이것에 뚫려
    # 고스트를 작업공간 밖으로 날려버렸다. 이 게이트가 그 사슬을 원천 차단한다.
    WORKSPACE_LO = np.array([0.05, -0.75, 0.0])
    WORKSPACE_HI = np.array([0.95, 0.60, 0.40])

    def __init__(self, min_score=0.5, ema_alpha=0.5, timeout=1.0,
                 tower_radius=0.12, tower_override=0.10, deadband=0.008,
                 freeze_radius=0.18, direct_write=True):
        self.min_score = float(min_score)
        self.ema_alpha = float(ema_alpha)
        self.timeout = float(timeout)
        self.tower_radius = float(tower_radius)
        self.tower_override = float(tower_override)
        self.deadband = float(deadband)
        # 엔드이펙터가 이 거리 안으로 들어온 블록은 갱신을 멈춘다 (조작 중으로 간주)
        self.freeze_radius = float(freeze_radius)
        self.direct_write = bool(direct_write)

        self._filtered = {}
        self.stats = {"published": 0, "gated_score": 0, "gated_tower": 0,
                      "override": 0, "attached": 0, "deadband": 0, "frozen": 0, "grasp_events": 0,
                      "gated_workspace": 0}
        self.last_reasons = {}

        self._held_name = None
        self._attach_offset = None
        self._last_width = None
        self._settle_count = 0

    # ------------------------------------------------------------------ 유틸
    def _ema(self, name, p):
        prev = self._filtered.get(name)
        if prev is None:
            self._filtered[name] = np.array(p, dtype=float)
        else:
            a = self.ema_alpha
            self._filtered[name] = a * np.asarray(p, dtype=float) + (1.0 - a) * prev
        return self._filtered[name]

    def reset(self):
        self._filtered.clear()
        self.last_reasons.clear()
        self._held_name = None
        self._attach_offset = None
        self._last_width = None
        self._settle_count = 0
        for k in self.stats:
            self.stats[k] = 0

    @staticmethod
    def _ee_pose(robot):
        try:
            T = np.asarray(robot.arm.get_fk_T(), dtype=float)
            return T[:3, :3], T[:3, 3]
        except Exception:
            return None, None

    # ------------------------------------------- 매 물리 스텝: 파지 추적 + 부착
    # 파지 판정은 논리 신호(in_gripper)가 아니라 **프로프리오셉션**으로 한다.
    # in_gripper 는 파지 시퀀스가 끝난 뒤에야 설정되고, 그마저도
    # monitor_gripper_has_block 이 belief 기준으로 검증해 깜빡인다(닭-달걀).
    # 반면 그리퍼 관절 폭은 물리 그 자체다:
    #   빈 채로 닫힘   -> 폭 ~0
    #   큐브를 쥠      -> 폭 ~0.0515 (큐브 한 변) 에서 멈춰 안정
    #   열림          -> 폭 ~0.08
    # "폭이 큐브 크기 근방에서 안정 + EE 가 어떤 고스트 바로 옆" = 물리적 파지 성립.
    GRASP_W_LO = CUBE_HALF * 2 - 0.010   # 0.0415
    GRASP_W_HI = CUBE_HALF * 2 + 0.007   # 0.0585
    RELEASE_W = CUBE_HALF * 2 + 0.010    # 이보다 열리면 놓은 것
    ATTACH_EE_RADIUS = 0.10              # 파지 성립 시 EE 에서 이 안에 있는 고스트를 부착
    SETTLE_STEPS = 5                     # 폭이 이만큼 연속 안정되어야 파지로 인정 (통과 중 오인 방지)

    def _gripper_width(self, robot):
        try:
            return float(robot.gripper.get_width())
        except Exception:
            try:
                return float(np.sum(robot.gripper.articulation_subset.get_joint_positions()))
            except Exception:
                return None

    def tick(self, context, robot):
        """**매 물리 스텝** 호출한다 (가볍다). 부착 중인 블록 이름을 반환."""
        blocks = getattr(context, "blocks", None)
        if not blocks:
            return None

        width = self._gripper_width(robot)
        R, t = self._ee_pose(robot)
        if width is None or R is None:
            return self._held_name

        # 폭 안정성 추적 (닫히는 중에 그립 창을 '통과'하는 것과 '멈춘' 것을 구분)
        if self._last_width is not None and abs(width - self._last_width) < 4e-4:
            self._settle_count += 1
        else:
            self._settle_count = 0
        self._last_width = width

        if self._held_name is None:
            grasped = (self.GRASP_W_LO < width < self.GRASP_W_HI
                       and self._settle_count >= self.SETTLE_STEPS)
            if grasped:
                # EE 에서 가장 가까운 고스트를 부착 후보로
                best, best_d = None, self.ATTACH_EE_RADIUS
                for name, block in blocks.items():
                    p, _ = block.obj.get_world_pose()
                    d = float(np.linalg.norm(np.asarray(p, dtype=float) - t))
                    if d < best_d:
                        best, best_d = name, d
                if best is not None:
                    block = blocks[best]
                    p, q = block.obj.get_world_pose()
                    # 파지 순간의 상대 자세를 **측정해서** 기록 (TCP 위치를 몰라도 정확)
                    self._attach_offset = (R.T @ (np.asarray(p, dtype=float) - t),
                                           np.asarray(q, dtype=float))
                    self._held_name = best
                    self.stats["grasp_events"] += 1
        else:
            if width > self.RELEASE_W:
                # 놓았다: 고스트는 마지막 부착 위치(= 배치 위치)에 그대로 남긴다.
                # EMA 잔존값(파지 전 테이블 위치)이 다음 발행에 섞이지 않도록 리셋.
                self._filtered.pop(self._held_name, None)
                self._held_name = None
                self._attach_offset = None

        if self._held_name is not None and self._attach_offset is not None:
            block = blocks[self._held_name]
            local_p, q0 = self._attach_offset
            p_att = t + R @ local_p
            block.obj.set_world_pose(position=p_att, orientation=q0)
            # measured 도 부착 위치로 갱신 - stale measured(파지 전 테이블 위치)가 남아 있으면
            # monitor_perception 의 15cm 분기가 고스트를 테이블로 되돌린다(실측 함정).
            self._set_measured(block, p_att, q0, time.time())
            self.stats["attached"] += 1

        return self._held_name

    # ------------------------------------------------------ 10Hz: 인식 발행
    def update(self, context, detections, estimates, robot):
        self.last_reasons = {}
        blocks = getattr(context, "blocks", None)
        if not blocks:
            return

        tower_pos = np.asarray(context.block_tower.tower_position, dtype=float)
        now = time.time()
        held_name = self._held_name
        _, ee_p = self._ee_pose(robot)

        if held_name:
            self.last_reasons[held_name] = "파지중 -> 그리퍼 부착"

        for cls, p in estimates.items():
            name = CLASS_TO_BLOCK.get(cls)
            if name is None or name not in blocks:
                continue
            if name == held_name:
                continue  # 부착이 담당

            det = detections.get(cls, {})
            if float(det.get("score", 0.0)) < self.min_score:
                self.stats["gated_score"] += 1
                self.last_reasons[name] = f"신뢰도 낮음 {det.get('score', 0):.2f}"
                continue

            # (1b) 작업공간 게이트 (클래스 상단 WORKSPACE 주석 참고)
            p_arr = np.asarray(p, dtype=float)
            if np.any(p_arr < self.WORKSPACE_LO) or np.any(p_arr > self.WORKSPACE_HI):
                self.stats["gated_workspace"] += 1
                self.last_reasons[name] = f"작업공간 밖 {np.round(p_arr, 2)} -> 기각"
                continue

            block = blocks[name]
            belief_p, q = block.obj.get_world_pose()
            belief_p = np.asarray(belief_p, dtype=float)

            # (4) 조작 중 동결 — 엔드이펙터가 붙은 블록은 목표가 흔들리면 안 된다.
            #     이걸 빼면 로봇이 큐브에 영원히 도달하지 못한다(실패 모드 (a)).
            if ee_p is not None and np.linalg.norm(belief_p - ee_p) < self.freeze_radius:
                self.stats["frozen"] += 1
                self.last_reasons[name] = "조작 중 -> belief 동결"
                continue

            p_f = self._ema(name, p).copy()
            p_f[2] = max(float(p_f[2]), CUBE_HALF)
            disagreement = float(np.linalg.norm(belief_p - p_f))

            # (3) 데드밴드
            if disagreement < self.deadband:
                self.stats["deadband"] += 1
                self.last_reasons[name] = f"변화 미미 {disagreement*1000:.1f}mm -> 유지"
                continue

            # (5) 탑 보호 — **예외 없음**. 탑 반경 안의 belief 는 조작이 완료한 결과다.
            # 처음엔 "인식이 크게 다르다고 하면 교정"하는 예외를 뒀지만, 실측에서 두 번
            # 연속으로 그 예외가 유령 검출에 뚫려 다 쌓은 블록의 belief 를 날렸다
            # (1차: 배경 깊이의 (-1.38,0.45) / 2차: 로봇 몸통 오검 (0.03,0.19) — 작업공간 안).
            # "조작이 확정한 belief 를 인식이 뒤집을 수 없다"가 원칙과도 일관된다.
            if self._is_in_tower(block, tower_pos):
                self.stats["gated_tower"] += 1
                self.last_reasons[name] = "탑에 적재됨 -> 보호"
                continue

            self._publish(block, p_f, np.asarray(q, dtype=float), now)
            self.stats["published"] += 1
            self.last_reasons.setdefault(name, "발행")

    def _set_measured(self, block, p, q, stamp):
        from isaacsim.cortex.framework.cortex_object import CortexMeasuredPose

        block.obj.set_measured_pose(CortexMeasuredPose(stamp, (p, q), self.timeout))

    def _publish(self, block, p, q, stamp):
        self._set_measured(block, p, q, stamp)
        # ghost 모드에서만 직접 쓴다. direct 모드(belief == 물리 큐브)에서 직접 쓰면
        # 큐브가 순간이동해 물리가 교란된다(실패 모드 (d)).
        if self.direct_write:
            block.obj.set_world_pose(position=p, orientation=q)

    def _is_in_tower(self, block, tower_pos):
        # xy 만 본다. 처음엔 z 조건(2층 이상)도 걸었지만 그러면 **1층 블록이 보호에서
        # 빠진다**. 탑 반경 안은 층수와 무관하게 조작이 관리하는 영역이다.
        p, _ = block.obj.get_world_pose()
        p = np.asarray(p, dtype=float)
        return np.linalg.norm(p[:2] - tower_pos[:2]) < self.tower_radius

    # ------------------------------------------------------------------ 보고
    def summary(self):
        s = self.stats
        return (f"발행 {s['published']} / 파지 {s['grasp_events']} / 부착 {s['attached']} / "
                f"동결 {s['frozen']} / 데드밴드 {s['deadband']} / 탑보호 {s['gated_tower']} / "
                f"작업공간기각 {s['gated_workspace']}")
