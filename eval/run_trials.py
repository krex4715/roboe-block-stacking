"""[ROBOE] 인식 기반 스태킹 배치 평가 - 인식 소스 x 배치 시나리오 매트릭스.

M6(랜덤 스폰 10회)에서 출발해, 4개 인식 소스 x 2개 배치 시나리오의 성공률 매트릭스로
확장했다. **한 프로세스에서 LOAD 를 반복**한다 (트라이얼마다 SimulationApp 재부팅 대비
~10초씩 절약). 이게 가능한 것 자체가 M2에서 고친 재로드 버그(월드 싱글톤 재사용)
수정의 실전 검증이다.

배치 시나리오 (--layout):
  - default: 과제 명세 그대로 - 큐브 4개가 일자 정위치. belief 도 같은 곳에서
    시작하므로 "인식이 유지·정밀화를 담당"하는 본연의 문제
  - random : 물리 큐브만 랜덤 위치·회전으로 옮기고 **고스트(belief)는 기본 스폰
    위치에 그대로 둔다.** 매 트라이얼이 "시작부터 belief 가 틀린 상태"에서 출발하고,
    인식이 그 오차를 스스로 교정해야만 성공한다. 인식이 죽어 있으면 로봇은 큐브가
    없는 기본 위치로 손을 뻗는다 - 성공률이 곧 인식 기여의 증명이다

인식 소스 (--backend): perception/detector_hub.py 의 BACKENDS 키.
  워커 백엔드(gdino/qwen)는 LOAD 마다 서브프로세스가 재기동되므로, **워커가 ready 를
  보고할 때까지 기다린 뒤 Start** 한다 - 로딩 시간이 트라이얼 시간을 잠식하면
  백엔드 간 비교가 불공정해진다. (로딩 시간은 CSV 의 worker_load_s 로 따로 기록)

랜덤 스폰 제약 (behavior 코드에서 역산):
  - 로봇 베이스에서 0.30~0.75 m (픽 유효 반경 0.25~0.81 에 여유)
  - 적재 위치에서 0.16 m 초과 (탑 보호/픽 거부 반경 밖)
  - 큐브 상호 간격 > 1.6변 (겹침 방지)
  - 카메라 시야 안 (SDG 와 같은 산란 영역)
같은 --seed 면 스폰 시퀀스가 같으므로, 백엔드만 바꿔 돌리면 **동일한 10개 배치**로
소스 간 성공률을 비교하게 된다 (통제 변인).

실행:
    python eval/run_trials.py --trials 10                                # 기본: finetuned/random
    python eval/run_trials.py --backend gdino --layout default --record
    bash eval/run_e2e_matrix.sh                                          # 8개 조합 전체
"""

import argparse
import csv
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=10)
parser.add_argument("--backend", choices=["finetuned", "yoloworld", "gdino", "qwen"],
                    default="finetuned", help="인식 소스 (perception/detector_hub.py BACKENDS)")
parser.add_argument("--layout", choices=["random", "default"], default="random",
                    help="default=명세 정위치 / random=랜덤 스폰+belief 오염")
parser.add_argument("--tower", type=float, nargs=2, default=[0.25, 0.30], metavar=("X", "Y"))
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max-steps", type=int, default=6000, help="트라이얼당 타임아웃 (물리 스텝, 완주는 ~1400)")
parser.add_argument("--record", action="store_true",
                    help="첫 트라이얼의 ZED 카메라 영상 저장 (원본 + 검출 오버레이 2벌)")
parser.add_argument("--out", type=str, default=None,
                    help="결과 폴더 (기본: media/e2e/<backend>_<layout>)")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import asyncio  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.examples.browser")
enable_extension("isaacsim.examples.interactive")
for _ in range(5):
    simulation_app.update()

from isaacsim.examples.interactive.user_examples.roboe_block_stacking.perception.detector import (  # noqa: E402
    select_best_per_class,
)
from isaacsim.examples.interactive.user_examples.roboe_block_stacking.roboe_stacking_example import (  # noqa: E402
    RoboeBlockStacking,
)
from isaacsim.examples.interactive.user_examples.roboe_block_stacking.scene_setup import (  # noqa: E402
    CUBE_HALF,
    CUBE_SIZE,
)

BEHAVIOR_PATH = str(REPO / "isaac_ext" / "roboe_block_stacking" / "behavior" / "block_stacking_behavior.py")
EXPECTED_ORDER = ["RedCube", "YellowCube", "GreenCube", "BlueCube"]
XY_TOL = 0.04
SCATTER_X, SCATTER_Y = (0.28, 0.72), (-0.55, 0.38)
MIN_GAP = CUBE_SIZE * 1.6
PHYSICS_HZ = 60.0  # sim_s = steps / PHYSICS_HZ. GPU 경합(느린 백엔드)과 무관한 시뮬 시간

outdir = Path(args.out) if args.out else REPO / "media" / "e2e" / f"{args.backend}_{args.layout}"
outdir.mkdir(parents=True, exist_ok=True)


def pump(coro, label, limit=6000):
    fut = asyncio.ensure_future(coro)
    for _ in range(limit):
        if fut.done():
            break
        simulation_app.update()
    if not fut.done():
        raise TimeoutError(label)
    fut.result()


def wait_detector_ready(sample, timeout_s=300.0):
    """워커 백엔드(gdino/qwen)의 모델 로딩 완료까지 대기. in-process 백엔드는 즉시 True.

    로딩 시간(초)을 돌려준다. 실패(워커 사망/타임아웃)면 RuntimeError - 준비 안 된
    백엔드로 트라이얼을 돌리면 전부 타임아웃 실패로 집계돼 데이터가 오염된다."""
    hub = sample.detector_hub
    if hub is None or hub.current_key is None:
        raise RuntimeError(f"검출기 미로드 (backend={args.backend}) - detector_hub 상태 확인")
    worker = hub._worker
    if worker is None:
        return 0.0
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if worker.dead:
            raise RuntimeError(f"워커 종료됨 - 로그: {worker.log_path}")
        if worker.ready:
            return round(worker.load_s or (time.time() - t0), 1)
        simulation_app.update()
    raise RuntimeError(f"워커 ready 타임아웃 {timeout_s}s - 로그: {worker.log_path}")


def sample_spawns(rng, tower_xy):
    """제약을 만족하는 랜덤 스폰 4개 (기각 샘플링)."""
    pts = []
    for _ in range(2000):
        if len(pts) == 4:
            break
        p = np.array([rng.uniform(*SCATTER_X), rng.uniform(*SCATTER_Y)])
        r = np.linalg.norm(p)
        # 파지 유효 작업공간: r >= 0.40 (실측 근거 - r~0.335 파지 시 팔이 접힌
        # 자세로 관절한계에 갇혀 RMPFlow 가 이탈 불가. diag14 트라이얼 6/10 재현.
        # 스톡 예제도 r 0.5~0.8 에서만 동작하도록 설계됨. 실제 셀의 작업 반경 명세와 동일 개념)
        if not (0.40 <= r <= 0.75):
            continue
        if np.linalg.norm(p - tower_xy) <= 0.16:
            continue
        if any(np.linalg.norm(p - q) <= MIN_GAP for q in pts):
            continue
        pts.append(p)
    while len(pts) < 4:  # 극히 드묾
        pts.append(np.array([0.30 + 0.13 * len(pts), -0.45]))
    return pts


def evaluate(cubes, tower_xy):
    rows = [(n, np.asarray(o.get_world_pose()[0], dtype=float)) for n, o in cubes.items()]
    stacked = sorted(rows, key=lambda r: r[1][2])
    order = [n for n, _ in stacked]
    max_off = max(float(np.linalg.norm(p[:2] - tower_xy)) for _, p in rows)
    return order == EXPECTED_ORDER and max_off <= XY_TOL, order, max_off


class TrialRecorder:
    """트라이얼 하나의 ZED 카메라 영상 2벌 - 원본(trialN_zed.mp4)과 검출 오버레이
    (trialN_zed_overlay.mp4). 오버레이는 GUI 검출 뷰와 같은 규칙(캡처 **후** 주석이라
    검출기로 되먹임 불가, 굵은 박스 = select_best 로 실제 채택된 검출)."""

    def __init__(self, outdir, trial):
        self.paths = {k: outdir / f"trial{trial}_zed{'_overlay' if k == 'overlay' else ''}.mp4"
                      for k in ("plain", "overlay")}
        self.writers = {}
        self.frames = 0

    def add(self, rgb, sample):
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2BGR)
        if not self.writers:
            h, w = bgr.shape[:2]
            self.writers = {k: cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
                            for k, p in self.paths.items()}
        self.writers["plain"].write(bgr)

        over = bgr.copy()
        dets = sample.last_detections
        best_ids = {id(d) for d in select_best_per_class(dets).values()}
        for d in dets:
            x0, y0, x1, y1 = (int(v) for v in d["box"])
            c = RoboeBlockStacking._DET_VIEW_COLORS.get(d["class"], (255, 255, 255))[::-1]  # RGB->BGR
            thick = 3 if id(d) in best_ids else 1
            cv2.rectangle(over, (x0, y0), (x1, y1), c, thick)
            cv2.putText(over, f"{d['class']} {d['score']:.2f}", (x0, max(y0 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
        if sample.detector_hub is not None:  # cv2 는 한글 불가 - ASCII 소스명만
            cv2.putText(over, f"source: {sample.detector_hub.current_short}",
                        (14, over.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
        self.writers["overlay"].write(over)
        self.frames += 1

    def close(self):
        for w in self.writers.values():
            w.release()
        if self.frames:
            print(f"    [record] {self.frames} 프레임 -> {self.paths['plain'].name} / "
                  f"{self.paths['overlay'].name}", flush=True)


def run_trial(sample, trial, tower_xy, spawn_spec):
    """트라이얼 1회: LOAD -> (랜덤이면 큐브 재배치) -> Start -> 완주/타임아웃 -> 채점."""
    t0 = time.time()
    pump(sample.load_world_async(), f"trial{trial}:LOAD")
    worker_load_s = wait_detector_ready(sample)

    if spawn_spec is not None:
        # 물리 큐브만 랜덤 배치. 고스트는 기본 위치 그대로 = belief 를 의도적으로 틀리게 시작.
        for (name, obj), (p, yaw) in zip(sample.cubes.items(), spawn_spec):
            obj.set_world_pose(position=np.array([p[0], p[1], CUBE_HALF]),
                               orientation=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
    # spawn_spec 이 None (default 배치): add_cubes 의 명세 위치 그대로. belief 도 같은 곳

    init_belief_err = [
        float(np.linalg.norm(np.asarray(sample.belief_cubes[n].get_world_pose()[0])[:2]
                             - np.asarray(sample.cubes[n].get_world_pose()[0])[:2]))
        for n in sample.cubes
    ]

    recorder = TrialRecorder(outdir, trial) if (args.record and trial == 0) else None
    try:
        pump(sample.on_event_async(), f"trial{trial}:Start")
        world = sample.get_world()
        ctx = sample.decider_network.context

        done_at = None
        for step in range(args.max_steps):
            world.step(render=True)
            if recorder is not None and step % 6 == 0 and sample.zed is not None:
                rgb, _ = sample.zed.capture()
                if rgb is not None:
                    recorder.add(rgb, sample)
            if ctx.block_tower.is_complete:
                if done_at is None:
                    done_at = step
                if step > done_at + 150:
                    break
    finally:
        if recorder is not None:
            recorder.close()

    ok, order, max_off = evaluate(sample.cubes, tower_xy)
    stats = dict(sample.bridge.stats)
    return {
        "trial": trial, "backend": args.backend, "layout": args.layout,
        "success": ok, "steps": step,
        "sim_s": round((done_at if done_at is not None else step) / PHYSICS_HZ, 1),
        "wall_s": round(time.time() - t0, 1), "worker_load_s": worker_load_s,
        "behavior_done": done_at is not None, "max_xy_off_mm": round(max_off * 1000, 1),
        "init_belief_err_mm": round(float(np.mean(init_belief_err)) * 1000, 1),
        "order": "->".join(order),
        "grasps": stats.get("grasp_events"), "ghost_rejects": stats.get("gated_workspace"),
        "published": stats.get("published"), "snapped": stats.get("snapped"),
        "reacquired": stats.get("reacquired"),
    }


def main():
    import traceback

    tower_xy = np.asarray(args.tower, dtype=float)
    rng = np.random.default_rng(args.seed)

    sample = RoboeBlockStacking(perception_fn=lambda t: None)
    sample.behavior = BEHAVIOR_PATH
    sample.belief_mode = "ghost"
    sample.detector_backend = args.backend
    sample.tower_position = np.array([tower_xy[0], tower_xy[1], 0.0])

    results = []
    for trial in range(args.trials):
        # 배치는 트라이얼 시작 전에 확정한다 - 재시도(아래)가 같은 배치를 재사용해야
        # "같은 seed = 같은 10개 배치" 통제 변인이 유지된다.
        spawn_spec = None
        if args.layout == "random":
            pts = sample_spawns(rng, tower_xy)
            spawn_spec = [(p, float(rng.uniform(0, np.pi / 2))) for p in pts]

        # Kit 비동기 엔진의 일회성 레이스("pop from an empty deque" 등)가 드물게
        # world.step 까지 전파돼 트라이얼을 죽인다 (스모크에서 1/4 확률로 실측).
        # 인식 성능과 무관한 인프라 소음이므로 같은 배치로 1회 재시도하고,
        # 두 번 연속 죽으면 order=ERROR 로 구분해 실패 집계한다 (조사 대상 표식).
        row = None
        for attempt in (0, 1):
            try:
                row = run_trial(sample, trial, tower_xy, spawn_spec)
                break
            except Exception as exc:
                print(f"[{args.backend}/{args.layout} trial {trial}] "
                      f"시도 {attempt + 1} 예외: {exc!r}", flush=True)
                traceback.print_exc()
        if row is None:
            row = {"trial": trial, "backend": args.backend, "layout": args.layout,
                   "success": False, "steps": -1, "sim_s": -1.0, "wall_s": -1.0,
                   "worker_load_s": -1.0, "behavior_done": False, "max_xy_off_mm": -1.0,
                   "init_belief_err_mm": -1.0, "order": "ERROR", "grasps": None,
                   "ghost_rejects": None, "published": None, "snapped": None,
                   "reacquired": None}
        results.append(row)

        # 트라이얼마다 즉시 저장 - 전체 타임아웃/정전에도 결과 보존
        csv_path = outdir / "trials.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        r = results[-1]
        print(f"[{args.backend}/{args.layout} trial {trial}] "
              f"{'PASS' if r['success'] else 'FAIL':4s} "
              f"sim={r['sim_s']}s wall={r['wall_s']}s "
              f"초기belief오차={r['init_belief_err_mm']}mm "
              f"파지={r['grasps']} 유령기각={r['ghost_rejects']}", flush=True)

    n_ok = sum(r["success"] for r in results)
    sims = [r["sim_s"] for r in results if r["success"]]
    print(f"\n[{args.backend}/{args.layout}] 성공률: {n_ok}/{len(results)} "
          f"({n_ok/len(results)*100:.0f}%)")
    if sims:
        print(f"완주 시뮬 시간: 평균 {np.mean(sims):.1f}s / 최소 {min(sims):.1f}s / 최대 {max(sims):.1f}s")
    print(f"초기 belief 오차(평균): {np.mean([r['init_belief_err_mm'] for r in results]):.0f}mm")
    print(f"결과 CSV: {outdir / 'trials.csv'}")
    print(f"BATCH_RESULT {args.backend}/{args.layout}: {n_ok}/{len(results)}")


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
    print(f"BATCH_RESULT {args.backend}/{args.layout}: FAIL (예외)")
    sys.stdout.flush()
    sys.stderr.flush()
finally:
    simulation_app.close()
