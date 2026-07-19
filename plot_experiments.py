"""生成所有实验对比图表：训练曲线 + 指标柱状图 + 类别分析"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── 全局样式 ──────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

ROOT = Path(__file__).parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── 色板：按 dataviz 原则，分类色固定顺序 ─────────────────
C = {
    "v8n":   "#3B82F6",  # blue
    "v8s":   "#6366F1",  # indigo
    "v11n":  "#10B981",  # emerald
    "v11s":  "#059669",  # green-dark
    "lr001": "#F59E0B",  # amber
    "ep50":  "#8B5CF6",  # violet
    "ep100": "#3B82F6",  # blue (same as v8n baseline)
    "ep150": "#EF4444",  # red
}

# ── 辅助 ──────────────────────────────────────────────────
def read_csv(name):
    """读取 results.csv，自动适配 ep50 的特殊目录"""
    path = ROOT / "runs" / "detect" / name / "results.csv"
    if not path.exists() and name == "exp_v8n_ep50":
        path = ROOT / "runs" / "detect" / "exp_v8n_ep50-3" / "results.csv"
    return pd.read_csv(path) if path.exists() else None

def load_results():
    return json.loads((ROOT / "experiment_results.json").read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════
# 图 1：训练曲线 — 三合一 (loss / mAP50 / mAP50-95)
# ══════════════════════════════════════════════════════════

def plot_training_curves(experiments, title, filename, color_map, legend_names=None):
    """为多组实验画三列训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    for name in experiments:
        df = read_csv(name)
        if df is None:
            continue
        c = color_map.get(name, "#999999")
        label = legend_names.get(name, name) if legend_names else name
        epochs = df["epoch"].values

        # Loss (train/box_loss + val/box_loss)
        ax = axes[0]
        ax.plot(epochs, df["train/box_loss"], color=c, lw=1.2, alpha=0.5, linestyle="--")
        ax.plot(epochs, df["val/box_loss"],   color=c, lw=1.8, label=label)

        # mAP@0.5
        ax = axes[1]
        ax.plot(epochs, df["metrics/mAP50(B)"], color=c, lw=1.8)

        # mAP@0.5:0.95
        ax = axes[2]
        ax.plot(epochs, df["metrics/mAP50-95(B)"], color=c, lw=1.8)

    # 样式统一
    for ax, ylabel in zip(axes, ["Box Loss", "mAP@0.5", "mAP@0.5:0.95"]):
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(fontsize=9, framealpha=0.8, loc="upper right")
    axes[0].set_title("Box Loss (train=虚线, val=实线)", fontsize=10, loc="left", pad=4)
    axes[1].set_title("Validation mAP@0.5", fontsize=10, loc="left", pad=4)
    axes[2].set_title("Validation mAP@0.5:0.95", fontsize=10, loc="left", pad=4)

    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, facecolor="white")
    plt.close(fig)
    print(f"  [OK] {filename}")


# ══════════════════════════════════════════════════════════
# 图 2：柱状图 — 模型对比（实验一）
# ══════════════════════════════════════════════════════════

def plot_exp1_model_comparison():
    """实验一：4 模型 mAP / 推理时间 / 模型大小对比"""
    data = {r["name"]: r for r in load_results()}
    models = ["exp_v8n", "exp_v8s", "exp_v11n", "exp_v11s"]
    labels = ["YOLOv8n", "YOLOv8s", "YOLOv11n", "YOLOv11s"]
    colors = [C["v8n"], C["v8s"], C["v11n"], C["v11s"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle("实验一：不同 YOLO 模型对比", fontsize=14, fontweight="bold")

    xs = np.arange(len(models))
    w = 0.55

    # (a) mAP
    ax = axes[0]
    m50  = [data[m]["mAP50"]     for m in models]
    m5095= [data[m]["mAP50_95"]  for m in models]
    ax.bar(xs - w/4, m50,   w/2, color=colors, edgecolor="white", lw=0.5, label="mAP@0.5")
    ax.bar(xs + w/4, m5095, w/2, color=colors, edgecolor="white", lw=0.5, alpha=0.45, label="mAP@0.5:0.95")
    for i in range(len(models)):
        ax.text(xs[i] - w/4, m50[i]   + 0.008, f"{m50[i]:.4f}",  ha="center", fontsize=8, fontweight="bold")
        ax.text(xs[i] + w/4, m5095[i] + 0.008, f"{m5095[i]:.4f}", ha="center", fontsize=8, color="#555555")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("mAP", fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # (b) 推理时间
    ax = axes[1]
    infs = [data[m]["inference_ms"] for m in models]
    ax.bar(xs, infs, w, color=colors, edgecolor="white", lw=0.5)
    for i, v in enumerate(infs):
        ax.text(i, v + 0.3, f"{v:.1f} ms", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Inference (ms)", fontsize=10)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # (c) 模型大小
    ax = axes[2]
    sizes = [data[m]["model_size_mb"] for m in models]
    ax.bar(xs, sizes, w, color=colors, edgecolor="white", lw=0.5)
    for i, v in enumerate(sizes):
        ax.text(i, v + 0.4, f"{v:.1f} MB", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Model Size (MB)", fontsize=10)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp1_model_comparison.png", facecolor="white")
    plt.close(fig)
    print("  [OK] exp1_model_comparison.png")


# ══════════════════════════════════════════════════════════
# 图 3：学习率对比（实验二）
# ══════════════════════════════════════════════════════════

def plot_exp2_lr_comparison():
    """实验二：lr0=0.01 vs 0.001"""
    data = {r["name"]: r for r in load_results()}
    experiments = ["exp_v8n", "exp_v8n_lr001"]
    labels = ["lr0=0.01\n(默认)", "lr0=0.001"]
    colors = [C["v8n"], C["lr001"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("实验二：学习率对比 (YOLOv8n)", fontsize=14, fontweight="bold")

    xs = np.arange(2); w = 0.5

    # (a) mAP
    ax = axes[0]
    m50  = [data[m]["mAP50"]    for m in experiments]
    m5095=[data[m]["mAP50_95"] for m in experiments]
    ax.bar(xs - w/4, m50,   w/2, color=colors, edgecolor="white", lw=0.5, label="mAP@0.5")
    ax.bar(xs + w/4, m5095, w/2, color=colors, edgecolor="white", lw=0.5, alpha=0.45, label="mAP@0.5:0.95")
    for i in range(2):
        ax.text(i - w/4, m50[i]   + 0.008, f"{m50[i]:.4f}",  ha="center", fontsize=9, fontweight="bold")
        ax.text(i + w/4, m5095[i] + 0.008, f"{m5095[i]:.4f}", ha="center", fontsize=9, color="#555555")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("mAP", fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # (b) 训练曲线 mAP@0.5
    ax = axes[1]
    for m, c, lb in zip(experiments, colors, ["lr0=0.01", "lr0=0.001"]):
        df = read_csv(m)
        if df is not None:
            ax.plot(df["epoch"], df["metrics/mAP50(B)"], color=c, lw=1.8, label=lb)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("mAP@0.5", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("训练过程 mAP@0.5 曲线", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp2_lr_comparison.png", facecolor="white")
    plt.close(fig)
    print("  [OK] exp2_lr_comparison.png")


# ══════════════════════════════════════════════════════════
# 图 4：训练轮次对比（实验三）
# ══════════════════════════════════════════════════════════

def plot_exp3_epochs_comparison():
    """实验三：50 / 100 / 150 epochs"""
    data = {r["name"]: r for r in load_results()}
    experiments = ["exp_v8n_ep50", "exp_v8n", "exp_v8n_ep150"]
    labels = ["50 epochs", "100 epochs\n(默认)", "150 epochs"]
    colors = [C["ep50"], C["v8n"], C["ep150"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("实验三：训练轮次对比 (YOLOv8n)", fontsize=14, fontweight="bold")

    xs = np.arange(3); w = 0.5

    # (a) mAP
    ax = axes[0]
    m50  = [data[m]["mAP50"]    for m in experiments]
    m5095=[data[m]["mAP50_95"] for m in experiments]
    ax.bar(xs - w/4, m50,   w/2, color=colors, edgecolor="white", lw=0.5, label="mAP@0.5")
    ax.bar(xs + w/4, m5095, w/2, color=colors, edgecolor="white", lw=0.5, alpha=0.45, label="mAP@0.5:0.95")
    for i in range(3):
        ax.text(i - w/4, m50[i]   + 0.008, f"{m50[i]:.4f}",  ha="center", fontsize=9, fontweight="bold")
        ax.text(i + w/4, m5095[i] + 0.008, f"{m5095[i]:.4f}", ha="center", fontsize=9, color="#555555")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("mAP", fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # (b) 训练曲线 mAP@0.5 overlay
    ax = axes[1]
    for m, c, lb in zip(experiments, colors, ["50 epochs", "100 epochs", "150 epochs"]):
        df = read_csv(m)
        if df is not None:
            ax.plot(df["epoch"], df["metrics/mAP50(B)"], color=c, lw=1.8, label=lb)
            # 标记终点
            ax.scatter(df["epoch"].iloc[-1], df["metrics/mAP50(B)"].iloc[-1],
                       color=c, s=30, zorder=5, edgecolors="white", lw=0.8)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("mAP@0.5", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("训练过程 mAP@0.5 曲线", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp3_epochs_comparison.png", facecolor="white")
    plt.close(fig)
    print("  [OK] exp3_epochs_comparison.png")


# ══════════════════════════════════════════════════════════
# 图 5：各类别 AP@0.5 对比（实验四 — 基于 v8n baseline）
# ══════════════════════════════════════════════════════════

def plot_per_class_ap():
    """实验四类别分析 + 所有模型各类别 AP 横向对比"""
    data = load_results()
    baseline = next(r for r in data if r["name"] == "exp_v8n")
    classes = list(baseline["per_class_ap50"].keys())
    ap_values = [baseline["per_class_ap50"][c] for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("各类别检测精度分析", fontsize=14, fontweight="bold")

    # (a) YOLOv8n baseline 各类别 AP
    ax = axes[0]
    bars = ax.barh(classes, ap_values, color="#3B82F6", edgecolor="white", lw=0.5, height=0.5)
    for bar, v in zip(bars, ap_values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("AP@0.5", fontsize=10)
    ax.set_title("YOLOv8n Baseline 各类别 AP@0.5", fontsize=11, loc="left")
    ax.grid(axis="x", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(ap_values) * 1.25)

    # (b) 所有模型各类别对比
    ax = axes[1]
    model_colors = {"exp_v8n": C["v8n"], "exp_v8s": C["v8s"],
                    "exp_v11n": C["v11n"], "exp_v11s": C["v11s"]}
    model_labels = {"exp_v8n": "v8n", "exp_v8s": "v8s",
                    "exp_v11n": "v11n", "exp_v11s": "v11s"}
    x = np.arange(len(classes))
    w = 0.2
    for i, (name, color) in enumerate(model_colors.items()):
        r = next(rr for rr in data if rr["name"] == name)
        vals = [r["per_class_ap50"].get(c, 0) for c in classes]
        ax.bar(x + i * w, vals, w, color=color, edgecolor="white", lw=0.3,
               label=model_labels[name])

    ax.set_xticks(x + w * 1.5); ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylabel("AP@0.5", fontsize=10)
    ax.set_title("四模型各类别 AP@0.5 横向对比", fontsize=11, loc="left")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp4_per_class_ap.png", facecolor="white")
    plt.close(fig)
    print("  [OK] exp4_per_class_ap.png")


# ══════════════════════════════════════════════════════════
# 图 6：综合汇总表
# ══════════════════════════════════════════════════════════

def plot_summary_table():
    """所有实验指标汇总为表格图"""
    data = load_results()
    rows = []
    for r in data:
        rows.append([
            r["desc"],
            r["model"],
            str(r.get("epochs", 100)),
            "[OK]" if r.get("augment", True) else "✗",
            str(r.get("lr0", r.get("name") == "exp_v8n_lr001" and 0.001 or 0.01)),
            f'{r["mAP50"]:.4f}',
            f'{r["mAP50_95"]:.4f}',
            f'{r.get("inference_ms", 0):.1f}',
            f'{r["model_size_mb"]:.1f}',
            f'{int(r.get("train_time_s", 0))}s',
        ])

    cols = ["实验", "模型", "Epochs", "增强", "lr0", "mAP@0.5", "mAP@0.5:0.95", "推理ms", "大小MB", "训练时间"]
    fig, ax = plt.subplots(figsize=(16, len(rows) * 0.55 + 1.2))
    ax.axis("off")
    fig.suptitle("全实验指标汇总", fontsize=13, fontweight="bold", y=1.01)

    table = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.6)

    # 浅色表头 + 斑马条纹
    for j, col in enumerate(cols):
        cell = table[0, j]
        cell.set_facecolor("#E5E7EB")
        cell.set_text_props(fontweight="bold")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if i % 2 == 0:
                table[i + 1, j].set_facecolor("#F9FAFB")

    fig.savefig(FIG_DIR / "summary_table.png", facecolor="white")
    plt.close(fig)
    print("  [OK] summary_table.png")


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    print("Generating experiment charts...")

    # ── 训练曲线：实验一（4 模型） ──
    plot_training_curves(
        experiments=["exp_v8n", "exp_v8s", "exp_v11n", "exp_v11s"],
        title="实验一训练曲线：不同 YOLO 模型 (mAP@0.5)",
        filename="training_exp1_models.png",
        color_map={"exp_v8n": C["v8n"], "exp_v8s": C["v8s"],
                   "exp_v11n": C["v11n"], "exp_v11s": C["v11s"]},
        legend_names={"exp_v8n": "v8n", "exp_v8s": "v8s",
                      "exp_v11n": "v11n", "exp_v11s": "v11s"},
    )

    # ── 训练曲线：实验二（学习率） ──
    plot_training_curves(
        experiments=["exp_v8n", "exp_v8n_lr001"],
        title="实验二训练曲线：学习率对比 (YOLOv8n)",
        filename="training_exp2_lr.png",
        color_map={"exp_v8n": C["v8n"], "exp_v8n_lr001": C["lr001"]},
        legend_names={"exp_v8n": "lr0=0.01", "exp_v8n_lr001": "lr0=0.001"},
    )

    # ── 训练曲线：实验三（epochs） ──
    plot_training_curves(
        experiments=["exp_v8n_ep50", "exp_v8n", "exp_v8n_ep150"],
        title="实验三训练曲线：训练轮次对比 (YOLOv8n)",
        filename="training_exp3_epochs.png",
        color_map={"exp_v8n_ep50": C["ep50"], "exp_v8n": C["v8n"],
                   "exp_v8n_ep150": C["ep150"]},
        legend_names={"exp_v8n_ep50": "50 epochs", "exp_v8n": "100 epochs",
                      "exp_v8n_ep150": "150 epochs"},
    )

    # ── 柱状图 ──
    plot_exp1_model_comparison()
    plot_exp2_lr_comparison()
    plot_exp3_epochs_comparison()
    plot_per_class_ap()
    plot_summary_table()

    print(f"\nAll done -> {FIG_DIR}/")
    print("  training_exp1_models.png   — 四模型训练曲线")
    print("  training_exp2_lr.png       — 学习率训练曲线")
    print("  training_exp3_epochs.png   — 轮次训练曲线")
    print("  exp1_model_comparison.png  — 模型对比柱状图")
    print("  exp2_lr_comparison.png     — 学习率对比")
    print("  exp3_epochs_comparison.png — 轮次对比")
    print("  exp4_per_class_ap.png      — 各类别精度分析")
    print("  summary_table.png          — 汇总表")


if __name__ == "__main__":
    main()
