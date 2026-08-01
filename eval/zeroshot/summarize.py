"""[ROBOE] zero-shot 비교 종합: 마크다운 표 출력 + 발표용 그림 생성.

실행: cd training && .venv/bin/python ../eval/zeroshot/summarize.py
출력: 표(stdout, README/슬라이드에 복붙용) + media/figures/zeroshot_compare.png
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, REPO, RESULTS  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (파일명, 표시명, 방식) - 표/그림 공용. yoloworld_green은 프롬프트 민감도 각주용.
ROWS = [
    ("finetuned_yolov8n", "YOLOv8n fine-tuned (SDG)", "폐쇄셋 학습"),
    ("gdino_tiny", "Grounding DINO (tiny)", "zero-shot (open-vocab)"),
    ("qwen25vl_3b", "Qwen2.5-VL-3B", "zero-shot (생성형 VLM)"),
    ("yoloworld_lightgreen", "YOLO-World v2 (s)", "zero-shot (open-vocab, 실시간)"),
]

reports = {name: json.loads((RESULTS / f"{name}.json").read_text()) for name, _, _ in ROWS}

print("| 모델 | 방식 | mAP50 | pick 정확도* | 지연/장 | 채점 |")
print("|---|---|---|---|---|---|")
for name, disp, method in ROWS:
    r = reports[name]
    print(f"| {disp} | {method} | {r['mAP50']:.3f} | "
          f"{r['policy_pick']['mean_pick_acc']:.3f} | {r['latency_ms']['mean']:.0f} ms | "
          f"{r['n_images']}장 |")
print("\n(*) pick 정확도 = 브리지 정책(클래스별 최고 신뢰도 1개 픽)이 실제 그 색 큐브에")
print("    맞은 프레임 비율 - 스태킹 성공을 가장 직접 예측하는 지표.")
gr = json.loads((RESULTS / "yoloworld_green.json").read_text())
print(f"프롬프트 민감도(YOLO-World): 연두를 'green cube'로 부르면 mAP50 {gr['mAP50']:.3f}, "
      f"'light green cube'로 부르면 {reports['yoloworld_lightgreen']['mAP50']:.3f}.")

# ---- 그림: (좌) 클래스별 AP50, (우) 정확도 vs 지연 트레이드오프 ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
disp_en = {
    "finetuned_yolov8n": "YOLOv8n fine-tuned\n(ours, SDG)",
    "gdino_tiny": "Grounding DINO\n(tiny)",
    "qwen25vl_3b": "Qwen2.5-VL-3B\n(generative VLM)",
    "yoloworld_lightgreen": "YOLO-World v2 (s)",
}
colors = {"finetuned_yolov8n": "#2b7de9", "gdino_tiny": "#e98a2b",
          "qwen25vl_3b": "#8a4fd3", "yoloworld_lightgreen": "#3aa657"}
bar_colors = ["#d94040", "#d9b83a", "#7ac943", "#3a6fd9"]  # 클래스 색 그대로

width = 0.2
for i, (name, _, _) in enumerate(ROWS):
    ap = reports[name]["ap50_per_class"]
    xs = [j + (i - 1.5) * width for j in range(len(CLASSES))]
    ax1.bar(xs, [ap[c] for c in CLASSES], width * 0.92,
            label=disp_en[name].replace("\n", " "),
            color=colors[name], alpha=0.35 + 0.65 * (name == "finetuned_yolov8n"))
ax1.set_xticks(range(len(CLASSES)))
ax1.set_xticklabels([c.replace("_cube", "") for c in CLASSES])
for t, c in zip(ax1.get_xticklabels(), bar_colors):
    t.set_color(c)
ax1.set_ylim(0, 1.05)
ax1.set_ylabel("AP50")
ax1.set_title("Per-class AP50 (same scorer, val 300)")
ax1.legend(fontsize=8, loc="lower right")
ax1.grid(axis="y", alpha=0.3)

for name, _, _ in ROWS:
    r = reports[name]
    ax2.scatter(r["latency_ms"]["mean"], r["mAP50"], s=140, color=colors[name], zorder=3)
    ax2.annotate(disp_en[name], (r["latency_ms"]["mean"], r["mAP50"]),
                 textcoords="offset points", xytext=(10, -4), fontsize=8.5)
ax2.axvline(100, color="gray", ls="--", lw=1)
ax2.text(100, 0.45, " 10 Hz loop budget", color="gray", fontsize=8.5, rotation=90, va="bottom")
ax2.set_xscale("log")
ax2.set_xlim(3, 20000)
ax2.set_ylim(0.4, 1.05)
ax2.set_xlabel("Latency per frame (ms, log)")
ax2.set_ylabel("mAP50")
ax2.set_title("Accuracy vs latency: zero-shot spectrum")
ax2.grid(alpha=0.3)

fig.tight_layout()
out = REPO / "media" / "figures" / "zeroshot_compare.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"\n그림 -> {out.relative_to(REPO)}")
