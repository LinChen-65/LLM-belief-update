import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

models = [
    "Qwen2.5-72B",
    "Qwen2.5-32B",
    "MiniMax-M2.5",
    "Gemini-2.5-Flash",
    "GPT-4o-mini",
    "GLM-4.7",
    "DeepSeek-V3"
]

fn_first = np.array([26.9, 19.0, 21.7, 8.0, 25.5, 30.9, 14.6])
fn_obs   = np.array([30.3, 35.1, 30.6, 38.6, 36.3, 33.3, 25.4])

fp_first = np.array([17.8, 25.7, 21.7, 37.5, 20.8, 13.4, 28.0])
fp_obs   = np.array([15.2, 11.5, 13.8, 8.4, 10.4, 11.1, 16.7])

y = np.arange(len(models))

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(
    1, 2,
    figsize=(22, 10.5),
    sharey=True,
    dpi=300
)

blue_first = "#0047cc"
blue_obs = "#2f80ed"
red_first = "#e60000"
red_obs = "#ff4b00"
line_color = "#4d4d4d"
grid_color = "#d9d9d9"

label_box = dict(
    boxstyle="round,pad=0.18",
    facecolor="white",
    edgecolor="none",
    alpha=0.82
)

def plot_panel(ax, first, obs, title, first_color, obs_color):
    for i in range(len(models)):
        ax.hlines(
            y=i,
            xmin=min(first[i], obs[i]),
            xmax=max(first[i], obs[i]),
            color=line_color,
            linewidth=2.4,
            alpha=0.85,
            zorder=1
        )

    ax.scatter(
        first, y,
        s=230,
        marker="o",
        color=first_color,
        edgecolor="white",
        linewidth=1.2,
        zorder=3
    )

    ax.scatter(
        obs, y,
        s=230,
        marker="D",
        color=obs_color,
        edgecolor="white",
        linewidth=1.2,
        zorder=3
    )

    # 数值标签放在哑铃两端两侧
    label_gap = 0.9

    for i, (first_val, obs_val) in enumerate(zip(first, obs)):
        if first_val <= obs_val:
            # first 在左，obs 在右
            ax.text(
                first_val - label_gap,
                y[i],
                f"{first_val:.1f}%",
                ha="right",
                va="center",
                color=first_color,
                fontsize=23,
                fontweight="bold",
                bbox=label_box,
                zorder=4,
                clip_on=False
            )
            ax.text(
                obs_val + label_gap,
                y[i],
                f"{obs_val:.1f}%",
                ha="left",
                va="center",
                color=obs_color,
                fontsize=23,
                fontweight="bold",
                bbox=label_box,
                zorder=4,
                clip_on=False
            )
        else:
            # obs 在左，first 在右
            ax.text(
                obs_val - label_gap,
                y[i],
                f"{obs_val:.1f}%",
                ha="right",
                va="center",
                color=obs_color,
                fontsize=23,
                fontweight="bold",
                bbox=label_box,
                zorder=4,
                clip_on=False
            )
            ax.text(
                first_val + label_gap,
                y[i],
                f"{first_val:.1f}%",
                ha="left",
                va="center",
                color=first_color,
                fontsize=23,
                fontweight="bold",
                bbox=label_box,
                zorder=4,
                clip_on=False
            )

    ax.set_title(title, fontsize=38, fontweight="bold", pad=28)
    ax.set_xlim(0, 44)
    ax.set_xticks(np.arange(0, 41, 5))

    ax.grid(axis="x", linestyle=(0, (4, 4)), color=grid_color, linewidth=1.4)
    ax.grid(axis="y", visible=False)

    ax.tick_params(axis="x", labelsize=26, width=1.8, length=9)
    ax.tick_params(axis="y", labelsize=28, width=1.8, length=9)

    ax.set_xlabel(
        "Percentage of Total Cases (%)",
        fontsize=29,
        labelpad=18
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)

plot_panel(
    axes[0],
    fn_first,
    fn_obs,
    "False Negative (FN)",
    blue_first,
    blue_obs
)

plot_panel(
    axes[1],
    fp_first,
    fp_obs,
    "False Positive (FP)",
    red_first,
    red_obs
)

axes[0].set_yticks(y)
axes[0].set_yticklabels(models, fontsize=30)
axes[0].invert_yaxis()

legend_handles = [
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=18,
        label="First-person"
    ),
    Line2D(
        [0], [0],
        marker="D",
        color="none",
        markerfacecolor="gray",
        markeredgecolor="gray",
        markersize=17,
        label="Observer"
    )
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=2,
    frameon=False,
    fontsize=31,
    bbox_to_anchor=(0.5, 1.025),
    columnspacing=2.8,
    handletextpad=0.9
)

plt.subplots_adjust(
    left=0.19,
    right=0.985,
    top=0.82,
    bottom=0.17,
    wspace=0.13
)

plt.savefig(
    "perspective_comparison_recreated_extra_large.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
