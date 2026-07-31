"""[ROBOE] Replicator 합성 데이터셋 생성 (YOLO 포맷).

**왜 합성 데이터인가**: 실제 라벨링 없이, 시뮬레이터가 정답 bbox를 자동으로 만들어준다.
게다가 배포 대상이 시뮬레이터 자체라 도메인 갭이 원리적으로 없다.
학습 이미지는 런타임과 **같은 코드(scene_setup)** 로 만든 씬을 **같은 카메라(ZED-X 좌안, 명세 pose)**
로 찍는다. 학습과 배포가 픽셀 단위로 같은 분포를 갖게 하는 것이 핵심.

**랜덤화 설계** (런타임 분포를 덮는 것이 목표):
  - 큐브 배치: 흩어짐 / 부분 탑(1~3층). 런타임에는 쌓는 중간 상태가 계속 나오므로 필수
  - 큐브 yaw: 0~90도 (정육면체는 90도 주기)
  - 로봇 자세: 관절 랜덤화 -> 팔이 큐브를 가리는 상황을 학습
  - 조명: 돔/디스턴트 세기·색·방향
  - 카메라: 미세 지터만 (+-2cm/+-2도). 런타임 카메라가 고정이라 시점 랜덤화는 불필요 -
    의도적 스코핑이다. 다만 실물 설치 오차를 흉내내는 정도는 남겨둔다

실행:
    python sdg/generate_dataset.py --train 2500 --val 300
    python sdg/generate_dataset.py --train 8 --val 2 --preview   # 스모크 + 시각 확인
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / "isaac_ext" / "roboe_block_stacking"

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=int, default=2500)
parser.add_argument("--val", type=int, default=300)
parser.add_argument("--out", type=str, default=str(REPO / "data" / "cubes"))
parser.add_argument("--subframes", type=int, default=12, help="프레임당 렌더 서브프레임 (노이즈 수렴)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resolution", type=int, nargs=2, default=[1280, 720])
parser.add_argument("--preview", action="store_true", help="bbox 그린 미리보기 이미지도 저장")
parser.add_argument("--gui", action="store_true")
parser.add_argument("--held-prob", type=float, default=0.0,
                    help="큐브 하나를 그리퍼 위치에 놓을 확률 (운반 중 장면 보충용)")
parser.add_argument("--start-index", type=int, default=0,
                    help="파일 이름 시작 번호 (보충 배치를 기존 데이터셋에 합칠 때 사용)")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.gui})

import carb  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.prims import get_prim_at_path  # noqa: E402
from isaacsim.core.utils.semantics import add_labels  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

sys.path.insert(0, str(EXT))
from perception.zed_camera import ZedXCamera  # noqa: E402
from scene_setup import (  # noqa: E402
    CUBE_HALF,
    CUBE_SIZE,
    CUBE_SPECS,
    add_cubes,
    add_lighting,
    add_tower_marker,
)

# 클래스 순서 = 과제의 쌓기 순서 (빨->노->연->파). 학습/추론이 같은 인덱스를 쓰도록 한 곳에서 정의.
CLASS_NAMES = ["red_cube", "yellow_cube", "green_cube", "blue_cube"]
PRIM_TO_CLASS = {"RedCube": "red_cube", "YellowCube": "yellow_cube",
                 "GreenCube": "green_cube", "BlueCube": "blue_cube"}

# 큐브를 흩뿌릴 영역 (카메라 시야 + 팔 도달 범위를 함께 만족하는 구간)
SCATTER_X = (0.28, 0.72)
SCATTER_Y = (-0.55, 0.38)
MIN_CUBE_GAP = CUBE_SIZE * 1.6  # 흩어짐 배치에서 큐브끼리 최소 간격

# Franka 관절: 기본 자세 주변으로 흔든다 (한계까지 흔들면 비현실적 자세가 많이 나온다)
FRANKA_HOME = np.array([0.0, -1.3, 0.0, -2.5, 0.0, 1.9, 0.75])
FRANKA_JITTER = np.array([1.2, 0.45, 1.0, 0.5, 1.2, 0.7, 1.5])
FRANKA_LIMITS_LO = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89])
FRANKA_LIMITS_HI = np.array([2.89, 1.76, 2.89, -0.07, 2.89, 3.75, 2.89])

OCCLUSION_DROP = 0.75  # 이 비율 넘게 가려진 박스는 라벨에서 제외
MIN_BOX_PX = 6         # 너무 작은 박스는 학습에 노이즈만 된다


def yaw_quat(yaw_rad):
    """z축 회전 쿼터니언 (w, x, y, z)."""
    return np.array([np.cos(yaw_rad / 2), 0.0, 0.0, np.sin(yaw_rad / 2)])


class Randomizer:
    """프레임마다 씬을 흔든다. 모든 무작위성은 이 클래스의 rng 하나에서 나온다(재현성)."""

    def __init__(self, cubes, franka, zed, seed, held_prob=0.0):
        self.cubes = cubes
        self.franka = franka
        self.zed = zed
        self.rng = np.random.default_rng(seed)
        self.names = [n for n, _ in CUBE_SPECS]
        self.zed_prim = get_prim_at_path(zed.prim_path)
        self.base_translate = np.array(zed.translate)
        self.base_rotate = np.array(zed.rotate_zyx)
        self.held_prob = float(held_prob)

    def _gripper_tcp(self):
        """그리퍼 파지 중심(TCP)의 월드 좌표. 없으면 None."""
        prim = get_prim_at_path("/World/Franka/panda_hand")
        if not prim or not prim.IsValid():
            return None
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pos = np.array(m.ExtractTranslation(), dtype=float)
        rot = np.array(m.ExtractRotationMatrix(), dtype=float)  # 행벡터 규약: 각 행이 기저축
        # 손 로컬 +z 방향으로 약 10cm 내려간 지점이 손가락 사이 파지 중심
        return pos + rot[2] * 0.10

    # ------------------------------------------------------------ 큐브 배치
    def _scatter_positions(self, n):
        """겹치지 않는 xy 를 n개 뽑는다 (기각 샘플링)."""
        pts = []
        for _ in range(400):
            if len(pts) == n:
                break
            p = np.array([self.rng.uniform(*SCATTER_X), self.rng.uniform(*SCATTER_Y)])
            if all(np.linalg.norm(p - q) > MIN_CUBE_GAP for q in pts):
                pts.append(p)
        while len(pts) < n:  # 드물게 실패하면 격자로 채운다
            pts.append(np.array([SCATTER_X[0] + 0.12 * len(pts), SCATTER_Y[0]]))
        return pts

    def randomize_cubes(self):
        """흩어짐(60%) 또는 부분 탑(40%). 런타임에는 쌓는 중간 상태가 계속 나온다."""
        order = list(self.names)
        self.rng.shuffle(order)
        tower_h = 0
        if self.rng.random() < 0.4:
            tower_h = int(self.rng.integers(1, 4))  # 1~3층

        tower_xy = np.array([self.rng.uniform(0.30, 0.60), self.rng.uniform(0.15, 0.40)])
        scatter = self._scatter_positions(len(order) - tower_h)

        si = 0
        for i, name in enumerate(order):
            if i < tower_h:
                xy = tower_xy
                z = (i + 0.5) * CUBE_SIZE + 0.001 * i
            else:
                xy = scatter[si]
                si += 1
                z = CUBE_HALF
            yaw = self.rng.uniform(0, np.pi / 2)  # 정육면체는 90도 주기
            self.cubes[name].set_world_pose(
                position=np.array([xy[0], xy[1], z]), orientation=yaw_quat(yaw)
            )
            self.cubes[name].set_linear_velocity(np.zeros(3))
            self.cubes[name].set_angular_velocity(np.zeros(3))

    # ------------------------------------------------------------ 로봇 자세
    def randomize_franka(self):
        if self.franka is None:
            return
        q = FRANKA_HOME + self.rng.uniform(-1, 1, 7) * FRANKA_JITTER
        q = np.clip(q, FRANKA_LIMITS_LO, FRANKA_LIMITS_HI)
        finger = self.rng.uniform(0.0, 0.04)
        full = np.concatenate([q, [finger, finger]])
        try:
            self.franka.set_joint_positions(full)
            self.franka.set_joint_velocities(np.zeros_like(full))
        except Exception as exc:  # 관절 수가 다르면 무시
            carb.log_warn(f"franka 관절 설정 실패: {exc}")

    # ------------------------------------------------------------ 조명
    def randomize_lighting(self):
        for path, base in (("/World/Lights/DomeLight", 500.0), ("/World/Lights/DistantLight", 700.0)):
            prim = get_prim_at_path(path)
            if not prim:
                continue
            intensity = base * self.rng.uniform(0.55, 1.6)
            prim.GetAttribute("inputs:intensity").Set(float(intensity))
            # 색온도 대신 RGB 를 살짝 흔든다 (따뜻한/차가운 조명 흉내).
            # 이게 노랑/연두 구분을 흔드는 가장 현실적인 교란이라 반드시 포함한다.
            tint = np.clip(1.0 + self.rng.uniform(-0.12, 0.12, 3), 0.0, 2.0)
            attr = prim.GetAttribute("inputs:color")
            if not attr:
                attr = prim.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f)
            attr.Set(Gf.Vec3f(*tint.astype(float)))
        # 태양 방향
        distant = get_prim_at_path("/World/Lights/DistantLight")
        if distant:
            xf = UsdGeom.Xformable(distant)
            xf.ClearXformOpOrder()
            xf.AddRotateXYZOp().Set(
                Gf.Vec3f(float(self.rng.uniform(-50, 10)), float(self.rng.uniform(20, 70)),
                         float(self.rng.uniform(-180, 180)))
            )

    # ------------------------------------------------------------ 카메라 지터
    def randomize_camera(self):
        """실물 설치 오차 수준의 미세 지터만. 런타임 카메라는 고정이므로 시점 랜덤화는 안 한다."""
        t = self.base_translate + self.rng.uniform(-0.02, 0.02, 3)
        r = self.base_rotate + self.rng.uniform(-2.0, 2.0, 3)
        self.zed_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*t.astype(float)))
        self.zed_prim.GetAttribute("xformOp:rotateZYX").Set(Gf.Vec3d(*r.astype(float)))

    def place_held_cube(self):
        """큐브 하나를 그리퍼 파지 위치로 옮긴다 (운반 중 장면).

        런타임에는 로봇이 큐브를 들고 이동하는 구간이 계속 나온다. 그때 큐브는
        공중에 뜨고(카메라에 더 크게 보이고) 손가락에 일부 가려진다.
        이 분포가 학습 데이터에 없으면 그 구간에서 검출이 불안정해진다.
        (관절을 먼저 세팅한 뒤 호출해야 TCP 가 맞는다)
        """
        if self.rng.random() >= self.held_prob:
            return
        tcp = self._gripper_tcp()
        if tcp is None or tcp[2] < CUBE_HALF:
            return
        name = self.names[int(self.rng.integers(0, len(self.names)))]
        self.cubes[name].set_world_pose(
            position=tcp, orientation=yaw_quat(self.rng.uniform(0, np.pi / 2))
        )
        self.cubes[name].set_linear_velocity(np.zeros(3))
        self.cubes[name].set_angular_velocity(np.zeros(3))

    def step(self):
        self.randomize_cubes()
        self.randomize_franka()
        self.randomize_lighting()
        self.randomize_camera()


def parse_bboxes(bbox_data, width, height):
    """annotator 출력 -> YOLO 라벨 줄 리스트."""
    data = bbox_data.get("data")
    info = bbox_data.get("info", {})
    id_to_labels = info.get("idToLabels", {}) or {}
    if data is None or len(data) == 0:
        return []

    def label_of(sem_id):
        v = id_to_labels.get(sem_id, id_to_labels.get(str(sem_id)))
        if isinstance(v, dict):
            v = v.get("class")
        return v

    lines = []
    for row in data:
        name = label_of(int(row["semanticId"]))
        if name not in CLASS_NAMES:
            continue
        if float(row["occlusionRatio"]) > OCCLUSION_DROP:
            continue
        x0, y0 = float(row["x_min"]), float(row["y_min"])
        x1, y1 = float(row["x_max"]), float(row["y_max"])
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(width - 1.0, x1), min(height - 1.0, y1)
        w, h = x1 - x0, y1 - y0
        if w < MIN_BOX_PX or h < MIN_BOX_PX:
            continue
        cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        lines.append((CLASS_NAMES.index(name), cx, cy, w / width, h / height, (x0, y0, x1, y1)))
    return lines


def draw_preview(rgb, labels, path):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    palette = [(60, 60, 220), (60, 220, 220), (60, 220, 120), (220, 100, 60)]
    for cls, _, _, _, _, (x0, y0, x1, y1) in labels:
        c = palette[cls % len(palette)]
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), c, 2)
        cv2.putText(img, CLASS_NAMES[cls], (int(x0), int(y0) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    cv2.imwrite(str(path), img)


def main():
    out = Path(args.out)
    width, height = args.resolution
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    if args.preview:
        (out / "preview").mkdir(parents=True, exist_ok=True)

    # Replicator 표준 설정 (공식 SDG 예제와 동일)
    rep.orchestrator.set_capture_on_play(False)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)  # Quality

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_lighting()
    cubes = add_cubes(world.scene)
    add_tower_marker(np.array([0.25, 0.30, 0.0]), world.scene)

    franka = None
    try:
        from isaacsim.cortex.framework.robot import add_franka_to_stage

        franka = world.scene.add(add_franka_to_stage(name="franka", prim_path="/World/Franka"))
    except Exception as exc:
        print(f"[sdg] Franka 추가 실패(계속 진행): {exc}")

    zed = ZedXCamera(resolution=(width, height))
    zed.spawn()

    # 시맨틱 라벨 - 이게 있어야 bbox annotator 가 클래스를 붙여준다
    for prim_name, cls in PRIM_TO_CLASS.items():
        add_labels(get_prim_at_path(f"/World/Obs/{prim_name}"), labels=[cls], instance_name="class")

    world.reset()

    rp = rep.create.render_product(zed.camera_prim_path, (width, height))
    rgb_annot = rep.annotators.get("rgb")
    rgb_annot.attach(rp)
    bbox_annot = rep.annotators.get("bounding_box_2d_tight")
    bbox_annot.attach(rp)

    randomizer = Randomizer(cubes, franka, zed, args.seed, held_prob=args.held_prob)

    counts = {c: 0 for c in CLASS_NAMES}
    stats = {"frames": 0, "boxes": 0, "empty_frames": 0}

    for split, n, seed_off in (("train", args.train, 0), ("val", args.val, 10_000)):
        if n <= 0:
            continue
        randomizer.rng = np.random.default_rng(args.seed + seed_off)  # val 은 다른 시드
        print(f"\n[sdg] {split} {n}장 생성 시작", flush=True)
        for i in range(n):
            randomizer.step()
            world.step(render=False)  # 물리에 반영 (관절/큐브 pose 전파)
            # 관절이 반영된 뒤라야 TCP 가 맞으므로 여기서 든 큐브를 배치하고 한 번 더 전파
            if randomizer.held_prob > 0:
                randomizer.place_held_cube()
                world.step(render=False)
            rep.orchestrator.step(rt_subframes=args.subframes)

            rgb = rgb_annot.get_data()
            bbox = bbox_annot.get_data()
            if rgb is None or len(rgb) == 0:
                continue
            rgb = np.asarray(rgb)[:, :, :3].astype(np.uint8)
            labels = parse_bboxes(bbox, width, height)

            stem = f"{args.start_index + i:06d}"
            cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(out / "labels" / split / f"{stem}.txt", "w") as f:
                for cls, cx, cy, w, h, _ in labels:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            if args.preview and i < 12:
                draw_preview(rgb, labels, out / "preview" / f"{split}_{stem}.jpg")

            stats["frames"] += 1
            stats["boxes"] += len(labels)
            if not labels:
                stats["empty_frames"] += 1
            for cls, *_ in labels:
                counts[CLASS_NAMES[cls]] += 1

            if (i + 1) % 50 == 0 or i + 1 == n:
                print(f"    {split} {i+1}/{n}  누적 박스 {stats['boxes']}", flush=True)

    # YOLO dataset.yaml - 클래스 순서를 여기서 고정해 학습/추론 인덱스 불일치를 막는다
    yaml_path = out / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {out}\ntrain: images/train\nval: images/val\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\nnames:\n")
        for i, n in enumerate(CLASS_NAMES):
            f.write(f"  {i}: {n}\n")

    meta = {
        "resolution": [width, height], "subframes": args.subframes, "seed": args.seed,
        "class_names": CLASS_NAMES, "per_class_boxes": counts, **stats,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"\n[sdg] 완료 - 프레임 {stats['frames']}, 박스 {stats['boxes']}, "
          f"빈 프레임 {stats['empty_frames']}")
    print(f"[sdg] 클래스별 박스 수: {counts}")
    print(f"[sdg] 출력: {out}")
    print(f"SDG_RESULT: {'PASS' if stats['frames'] > 0 and stats['boxes'] > 0 else 'FAIL'}")


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
    print("SDG_RESULT: FAIL")
    sys.stdout.flush()
    sys.stderr.flush()
finally:
    simulation_app.close()
