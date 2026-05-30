import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 数据
# =========================
models = [
    "DeepSeek-V3",
    "Gemini-2.5-Flash",
    "GPT-4o-mini",
    "MiniMax-M2.5",
    "GLM-4.7",
    "Qwen-32B",
    "Qwen-72B"
]

kappa_matrix = np.array([
    [1.000, 0.426, 0.424, 0.456, 0.387, 0.528, 0.453],
    [0.426, 1.000, 0.284, 0.264, 0.190, 0.320, 0.243],
    [0.424, 0.284, 1.000, 0.352, 0.436, 0.470, 0.533],
    [0.456, 0.264, 0.352, 1.000, 0.426, 0.398, 0.421],
    [0.387, 0.190, 0.436, 0.426, 1.000, 0.401, 0.548],
    [0.528, 0.320, 0.470, 0.398, 0.401, 1.000, 0.511],
    [0.453, 0.243, 0.533, 0.421, 0.548, 0.511, 1.000],
])

# =========================
# 2. 只保留下三角
# =========================
mask = np.triu(np.ones_like(kappa_matrix, dtype=bool), k=1)
masked_matrix = np.ma.masked_where(mask, kappa_matrix)

# =========================
# 3. 绘图参数
# =========================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

cmap = plt.cm.RdBu_r.copy()
cmap.set_bad(color="white")

im = ax.imshow(
    masked_matrix,
    cmap=cmap,
    vmin=-1,
    vmax=1,
    interpolation="nearest"
)

# =========================
# 4. 坐标轴
# =========================
ax.set_xticks(np.arange(len(models)))
ax.set_yticks(np.arange(len(models)))

ax.set_xticklabels(
    models,
    rotation=50,
    ha="right",
    fontsize=30
)

ax.set_yticklabels(
    models,
    fontsize=30
)

ax.tick_params(axis="both", length=0)

# =========================
# 5. 数值标注
# =========================
for i in range(len(models)):
    for j in range(len(models)):
        if j <= i:
            value = kappa_matrix[i, j]
            text_color = "white" if value >= 0.45 else "black"

            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=28,
                color=text_color,
                fontweight="normal"
            )

# =========================
# 6. 去掉边框
# =========================
for spine in ax.spines.values():
    spine.set_visible(False)

# =========================
# 7. Colorbar
# =========================
cbar = plt.colorbar(
    im,
    ax=ax,
    fraction=0.052,
    pad=0.06
)

cbar.set_label(
    "Cohen's κ",
    fontsize=32,
    labelpad=22
)

cbar.ax.tick_params(labelsize=28)

# =========================
# 8. 布局与保存
# =========================
plt.tight_layout()

plt.savefig(
    "pairwise_cohens_kappa_lower_triangle_large_font.png",
    dpi=300,
    bbox_inches="tight")

plt.show()
