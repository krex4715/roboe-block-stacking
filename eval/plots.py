"""[ROBOE] 발표용 그림 생성 (실측 CSV 에서 직접 계산 - 수치 하드코딩 금지).

생성물 (media/figures/):
    ablation_depth_correction.png  깊이->중심 보정 3종의 3D 오차 (M4 실측 120건)
    trials_summary.png             M6 배치 평가 요약 (trials.csv 존재 시)

실행 (isaacsim 환경, GPU 불필요):
    python eval/plots.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "media" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# dataviz 검증 팔레트 (측정값은 잉크색 텍스트, 색은 마크에만)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
PRIMARY = "#2a78d6"   # 채택안 강조
MUTED = "#c3c2b7"     # 비채택안
GRID = "#e8e7e2"

# 한글 폰트 (Noto Sans CJK - .ttc 는 rcParams 로는 안 잡히는 경우가 있어 직접 등록)
from matplotlib import font_manager

for _fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if Path(_fp).exists():
        font_manager.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "Noto Sans CJK JP"  # CJK 통합 폰트 - 한글 글리프 포함
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
})


def fig_ablation():
    path = REPO / "media" / "m4" / "perception_error.csv"
    rows = [r for r in csv.DictReader(open(path)) if r.get("status") == "ok"]
    if not rows:
        print("ablation: 데이터 없음")
        return

    methods = [
        ("none", "보정 없음\n(표면점 그대로)"),
        ("ray", "반큐브 오프셋"),
        ("box", "광선-박스 정확해\n(채택)"),
    ]
    means = [np.mean([float(r[f"err_{m}"]) for r in rows]) * 1000 for m, _ in methods]
    p95s = [np.percentile([float(r[f"err_{m}"]) for r in rows], 95) * 1000 for m, _ in methods]

    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=200)
    y = np.arange(len(methods))[::-1]
    colors = [MUTED, MUTED, PRIMARY]
    bars = ax.barh(y, means, height=0.52, color=colors, zorder=3)
    for yi, bar, mean, p95 in zip(y, bars, means, p95s):
        ax.plot([p95], [yi], marker="|", ms=16, mew=2, color=INK2, zorder=4)
        ax.text(bar.get_width() + 0.45, yi, f"{mean:.1f} mm",
                va="center", color=INK, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in methods])
    ax.set_xlabel("큐브 중심 3D 오차 (mm)   |  = p95     n=%d (YOLO 검출 결합, 정지 큐브)" % len(rows))
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("깊이는 '앞면'을 준다 — 중심 보정별 오차", loc="left", fontweight="bold", pad=12)
    # 큐브 절반 크기 기준선: 이보다 크면 파지 실패권
    half = 25.75
    ax.axvline(half, color=INK2, lw=1, ls="--", zorder=2)
    ax.text(half + 0.3, y[0] + 0.42, "반큐브 25.8mm\n(파지 실패권)", fontsize=9.5, color=INK2)

    fig.tight_layout()
    fig.savefig(OUT / "ablation_depth_correction.png", bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {OUT/'ablation_depth_correction.png'}  (means={np.round(means,1)})")


def fig_trials():
    path = REPO / "media" / "m6" / "trials.csv"
    if not path.exists():
        print("trials: 아직 없음 (M6 후 재실행)")
        return
    rows = list(csv.DictReader(open(path)))
    ok = [r for r in rows if r["success"] == "True"]
    walls = [float(r["wall_s"]) for r in ok]
    init_err = [float(r["init_belief_err_mm"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=200)

    ax = axes[0]
    n = np.arange(1, len(rows) + 1)
    colors = [PRIMARY if r["success"] == "True" else "#d95926" for r in rows]
    ax.bar(n, [float(r["wall_s"]) for r in rows], color=colors, width=0.62, zorder=3)
    ax.set_xticks(n)
    ax.set_xlabel("트라이얼 (랜덤 스폰)")
    ax.set_ylabel("완주 시간 (s)")
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_title(f"성공 {len(ok)}/{len(rows)} — 평균 {np.mean(walls):.1f}s" if walls
                 else f"성공 {len(ok)}/{len(rows)}", loc="left", fontweight="bold")

    ax = axes[1]
    ax.bar(n, init_err, color=MUTED, width=0.62, zorder=3)
    ax.set_xticks(n)
    ax.set_xlabel("트라이얼")
    ax.set_ylabel("시작 시 belief 오차 (mm)")
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_title("매 트라이얼, 인식이 스스로 교정한 초기 오차", loc="left", fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "trials_summary.png", bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {OUT/'trials_summary.png'}")


fig_ablation()
fig_trials()
