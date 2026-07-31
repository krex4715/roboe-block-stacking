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
from collections import Counter, deque

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

    def __init__(self, min_score=0.4, ema_alpha=0.5, timeout=1.0,
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
                      "override": 0, "attached": 0, "deadband": 0, "frozen": 0}
        self.last_reasons = {}

        self._grip_votes = deque(maxlen=20)   # 매 물리 스텝 투표 -> 약 1/3초 창
        self._held_name = None
        self._attach_offset = None

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
        self._grip_votes.clear()
        self._held_name = None
        self._attach_offset = None
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
    def tick(self, context, robot):
        """**매 물리 스텝** 호출한다 (가볍다).

        `in_gripper` 는 매 스텝 깜빡이므로 여기서 촘촘히 투표해야 안정된 판정이 나온다.
        인식 주기(10Hz)로 샘플링하면 엇박이 나 부착이 발동하지 않는다(실패 모드 (c)).
        """
        blocks = getattr(context, "blocks", None)
        if not blocks:
            return None

        cur = getattr(context, "in_gripper", None)
        self._grip_votes.append(cur.name if cur is not None else None)
        counts = Counter(v for v in self._grip_votes if v)
        held = None
        if counts:
            name, n = counts.most_common(1)[0]
            if n > len(self._grip_votes) * 0.3:
                held = name

        if held != self._held_name:
            self._held_name = held
            self._attach_offset = None
        if held is None or held not in blocks:
            return None

        R, t = self._ee_pose(robot)
        if R is None:
            return held
        block = blocks[held]
        p, q = block.obj.get_world_pose()

        if self._attach_offset is None:
            # 파지 순간의 상대 자세를 **측정해서** 기록. TCP 오프셋을 몰라도 정확해지는 이유.
            self._attach_offset = (R.T @ (np.asarray(p, dtype=float) - t),
                                   np.asarray(q, dtype=float))
            return held

        local_p, q0 = self._attach_offset
        block.obj.set_world_pose(position=t + R @ local_p, orientation=q0)
        self.stats["attached"] += 1
        return held

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

            # (5) 탑 보호 (단, 크게 어긋나면 교정)
            if self._is_in_tower(block, tower_pos):
                if disagreement < self.tower_override:
                    self.stats["gated_tower"] += 1
                    self.last_reasons[name] = "탑에 적재됨 -> 보호"
                    continue
                self.stats["override"] += 1
                self.last_reasons[name] = f"탑이지만 불일치 {disagreement*100:.0f}cm -> 교정"

            self._publish(block, p_f, np.asarray(q, dtype=float), now)
            self.stats["published"] += 1
            self.last_reasons.setdefault(name, "발행")

    def _publish(self, block, p, q, stamp):
        from isaacsim.cortex.framework.cortex_object import CortexMeasuredPose

        block.obj.set_measured_pose(CortexMeasuredPose(stamp, (p, q), self.timeout))
        # ghost 모드에서만 직접 쓴다. direct 모드(belief == 물리 큐브)에서 직접 쓰면
        # 큐브가 순간이동해 물리가 교란된다(실패 모드 (d)).
        if self.direct_write:
            block.obj.set_world_pose(position=p, orientation=q)

    def _is_in_tower(self, block, tower_pos):
        p, _ = block.obj.get_world_pose()
        p = np.asarray(p, dtype=float)
        return (np.linalg.norm(p[:2] - tower_pos[:2]) < self.tower_radius
                and p[2] > CUBE_HALF * 2.0)

    # ------------------------------------------------------------------ 보고
    def summary(self):
        s = self.stats
        return (f"발행 {s['published']} / 부착 {s['attached']} / 동결 {s['frozen']} / "
                f"데드밴드 {s['deadband']} / 탑보호 {s['gated_tower']}")
