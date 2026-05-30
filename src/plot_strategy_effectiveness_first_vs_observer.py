import matplotlib.pyplot as plt
import numpy as np

# =========================
# 数据部分：按你的真实数据替换即可
# =========================
strategies = ["Pathos", "Ethos", "E+P", "Logos", "L+P", "L+E", "L+E+P"]

human_mean = {
    "Pathos": 0.41,
    "Ethos": 0.31,
    "E+P": 0.43,
    "Logos": 0.45,
    "L+P": 0.50,
    "L+E": 0.56,
    "L+E+P": 0.57
}

llm_mean = {
    "Pathos": 0.23,
    "Ethos": 0.30,
    "E+P": 0.48,
    "Logos": 0.47,
    "L+P": 0.54,
    "L+E": 0.58,
    "L+E+P": 0.63
}

llm_models = {
    "Pathos": [0.10, 0.13, 0.17, 0.20, 0.24, 0.27, 0.45],
    "Ethos": [0.23, 0.30, 0.31, 0.54],
    "E+P": [0.31, 0.38, 0.42, 0.47, 0.49, 0.50, 0.74],
    "Logos": [0.28, 0.34, 0.40, 0.44, 0.46, 0.52, 0.56, 0.76],
    "L+P": [0.32, 0.41, 0.48, 0.58, 0.65, 0.81],
    "L+E": [0.39, 0.46, 0.57, 0.59, 0.70, 0.85],
    "L+E+P": [0.41, 0.54, 0.55, 0.67, 0.79, 0.85]
}

colors = {
    "Pathos": "#8c5a44",
    "Ethos": "#ef9aa4",
    "E+P": "#7a52c7",
    "Logos": "#2d7bd8",
    "L+P": "#f39c12",
    "L+E": "#27ae60",
    "L+E+P": "#e74c3c"
}

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(9, 8.5), dpi=300)

# =========================
# 背景区域
# =========================
x_min, x_max = 0.15, 0.66
y_min, y_max = 0.08, 0.88

x_fill = np.linspace(x_min, x_max, 500)

ax.fill_between(
    x_fill,
    x_fill,
    y_max,
    color="#dfe6ec",
    alpha=0.85,
    zorder=0
)

ax.fill_between(
    x_fill,
    y_min,
    x_fill,
    color="#efe8e8",
    alpha=0.85,
    zorder=0
)

# 对角线
ax.plot(
    [x_min, x_max],
    [x_min, x_max],
    linestyle="--",
    color="0.6",
    linewidth=1.4,
    zorder=1
)

# 网格
ax.grid(
    True,
    linestyle="--",
    linewidth=1.0,
    alpha=0.28
)

# =========================
# 单模型点：同一颜色固定在同一条竖线上
# =========================
for s in strategies:
    x_same = np.full(len(llm_models[s]), human_mean[s])

    ax.scatter(
        x_same,
        llm_models[s],
        s=52,
        color=colors[s],
        alpha=0.45,
        edgecolors="white",
        linewidth=0.6,
        zorder=2
    )

# =========================
# 均值菱形：放大
# =========================
for s in strategies:
    ax.scatter(
        human_mean[s],
        llm_mean[s],
        marker="D",
        s=260,
        color=colors[s],
        edgecolor="white",
        linewidth=2.2,
        zorder=5
    )

# =========================
# 策略文字：放大
# =========================
label_offsets = {
    "Pathos": (0.008, 0.010),
    "Ethos": (0.008, 0.006),
    "E+P": (0.008, 0.006),
    "Logos": (0.008, -0.006),
    "L+P": (0.008, 0.006),
    "L+E": (0.008, 0.004),
    "L+E+P": (0.008, 0.006)
}

for s in strategies:
    dx, dy = label_offsets[s]
    ax.text(
        human_mean[s] + dx,
        llm_mean[s] + dy,
        s,
        fontsize=22,
        fontweight="bold",
        color=colors[s],
        zorder=6
    )

# =========================
# 区域文字
# =========================
ax.text(
    x_min + 0.02,
    y_max - 0.035,
    "LLM > Human",
    fontsize=22,
    fontweight="bold",
    color="#6e9ad0"
)

ax.text(
    x_max - 0.14,
    y_min + 0.025,
    "LLM < Human",
    fontsize=22,
    fontweight="bold",
    color="#d97f7f"
)

# 删除右下角灰色文字：
# 不再添加下面这类说明
# ax.text(..., "◆ = mean across models", ...)
# ax.text(..., "● = individual model", ...)

# =========================
# 坐标轴设置：收紧范围
# =========================
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ax.set_xlabel("Persuasion Rate (Human)", fontsize=24, labelpad=12)
ax.set_ylabel("Persuasion Rate (First-person)", fontsize=24, labelpad=12)

ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_yticks([0.2, 0.4, 0.6, 0.8])

ax.tick_params(axis="both", labelsize=18, width=1.4, length=6)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.subplots_adjust(
    left=0.13,
    right=0.98,
    bottom=0.12,
    top=0.98
)

plt.savefig(
    "persuasion_strategy_aligned_clean.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
