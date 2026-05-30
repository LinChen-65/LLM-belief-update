import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# =========================
# 1. 文件路径
# =========================
AGENT_TXT = "/data7/chenyitong/Winning_Arguments/final_v3_Experiment_Analysis_Results/agent_nine_mechanism_feature_logistic_regression/agent_logistic_regression_model_summary.txt"
OBSERVER_TXT = "/data7/chenyitong/Winning_Arguments/final_v3_Experiment_Analysis_Results/observer_nine_mechanism_feature_logistic_regression/observer_logistic_regression_model_summary.txt"
HUMAN_TXT = "/data7/chenyitong/Winning_Arguments/final_v3_Experiment_Analysis_Results/human_nine_mechanism_feature_logistic_regression/human_logistic_regression_model_summary.txt"

# 如果你在服务器上运行，可以改成绝对路径，例如：
# AGENT_TXT = "/data7/chenyitong/Winning_Arguments/agent_logistic_regression_model_summary.txt"
# OBSERVER_TXT = "/data7/chenyitong/Winning_Arguments/observer_logistic_regression_model_summary.txt"
# HUMAN_TXT = "/data7/chenyitong/Winning_Arguments/human_logistic_regression_model_summary.txt"

# =========================
# 2. feature 名称映射
# =========================
FEATURE_LABELS = {
    "z_op_length": "OP Length",
    "z_op_def_freq": "OP Def Freq",
    "z_op_i_freq": "OP I Freq",
    "z_ch_we_freq": "Ch We Freq",
    "z_ch_has_formatting": "Ch Has Formatting",
    "z_ch_def_freq": "Ch Def Freq",
    "z_ch_dissimilarity": "Ch Dissimilarity",
    "z_ch_has_link": "Ch Has Link",
    "z_ch_length": "Ch Length",
}

FEATURES = list(FEATURE_LABELS.keys())

# =========================
# 3. 读取并解析 statsmodels summary txt
# =========================
def parse_glm_summary_txt(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")

    rows = []

    # 匹配形如：
    # z_ch_length 0.3814 0.033 11.589 0.000 0.317 0.446
    pattern = re.compile(
        r"^(z_[A-Za-z0-9_]+)\s+"
        r"([-+]?\d*\.\d+|[-+]?\d+)\s+"
        r"([-+]?\d*\.\d+|[-+]?\d+)\s+"
        r"([-+]?\d*\.\d+|[-+]?\d+)\s+"
        r"(<\s*)?([-+]?\d*\.\d+|[-+]?\d+)\s+"
        r"([-+]?\d*\.\d+|[-+]?\d+)\s+"
        r"([-+]?\d*\.\d+|[-+]?\d+)"
    )

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = pattern.match(line)
            if not m:
                continue

            feature = m.group(1)
            if feature not in FEATURES:
                continue

            coef = float(m.group(2))
            std_err = float(m.group(3))
            z_value = float(m.group(4))

            p_prefix = m.group(5) or ""
            p_value_raw = m.group(6)
            p_text = f"< {p_value_raw}" if p_prefix.strip() == "<" else p_value_raw

            ci_low = float(m.group(7))
            ci_high = float(m.group(8))

            rows.append({
                "feature": feature,
                "coef": coef,
                "std_err": std_err,
                "z": z_value,
                "p_text": p_text,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "OR": np.exp(coef),
                "OR_ci_low": np.exp(ci_low),
                "OR_ci_high": np.exp(ci_high),
            })

    df = pd.DataFrame(rows)

    missing = sorted(set(FEATURES) - set(df["feature"]))
    if missing:
        print(f"警告：{path.name} 中未找到这些 feature:")
        for item in missing:
            print(" -", item)

    return df


agent_df = parse_glm_summary_txt(AGENT_TXT)
observer_df = parse_glm_summary_txt(OBSERVER_TXT)
human_df = parse_glm_summary_txt(HUMAN_TXT)

# =========================
# 4. 合并三类结果
# =========================
plot_df = pd.DataFrame({"feature": FEATURES})

plot_df = plot_df.merge(
    agent_df[["feature", "OR", "coef", "p_text"]],
    on="feature",
    how="left"
).rename(columns={
    "OR": "agent_OR",
    "coef": "agent_coef",
    "p_text": "agent_p"
})

plot_df = plot_df.merge(
    observer_df[["feature", "OR", "coef", "p_text"]],
    on="feature",
    how="left"
).rename(columns={
    "OR": "observer_OR",
    "coef": "observer_coef",
    "p_text": "observer_p"
})

plot_df = plot_df.merge(
    human_df[["feature", "OR", "coef", "p_text"]],
    on="feature",
    how="left"
).rename(columns={
    "OR": "human_OR",
    "coef": "human_coef",
    "p_text": "human_p"
})

plot_df["label"] = plot_df["feature"].map(FEATURE_LABELS)

# 按 Human OR 从小到大排序，使图形顺序接近你给的示例
plot_df = plot_df.sort_values("human_OR", ascending=True).reset_index(drop=True)

# 保存一份提取后的数据，方便检查
plot_df.to_csv("mechanism_feature_or_extracted.csv", index=False, encoding="utf-8-sig")

# =========================
# 5. 绘图
# =========================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(18, 11), dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

y = np.arange(len(plot_df))

agent_color = "#ef2b2d"
observer_color = "#00a878"
human_color = "#2f80ed"
line_color = "#555555"
grid_color = "#d8d8d8"

# 灰色连接线：连接红色 LLM first-person 与绿色 LLM observer
for i, row in plot_df.iterrows():
    x1 = row["agent_OR"]
    x2 = row["observer_OR"]

    ax.hlines(
        y=i,
        xmin=min(x1, x2),
        xmax=max(x1, x2),
        color=line_color,
        linewidth=3.0,
        alpha=0.72,
        zorder=1
    )

# 红色菱形：LLM First-person
ax.scatter(
    plot_df["agent_OR"],
    y,
    s=420,
    marker="D",
    color=agent_color,
    edgecolor="white",
    linewidth=1.4,
    zorder=4,
    label="LLM (First-person)"
)

# 绿色方块：LLM Observer
ax.scatter(
    plot_df["observer_OR"],
    y,
    s=420,
    marker="s",
    color=observer_color,
    edgecolor="white",
    linewidth=1.4,
    zorder=4,
    label="LLM (Observer)"
)

# 蓝色竖线：Human Reference
for i, row in plot_df.iterrows():
    ax.vlines(
        x=row["human_OR"],
        ymin=i - 0.20,
        ymax=i + 0.20,
        color=human_color,
        linewidth=4.0,
        zorder=5
    )

# OR = 1 参考线
ax.axvline(
    1.0,
    color="#808080",
    linestyle=(0, (5, 5)),
    linewidth=2.0,
    alpha=0.9,
    zorder=0
)

# y 轴 feature 名
ax.set_yticks(y)
ax.set_yticklabels(plot_df["label"], fontsize=34)
ax.invert_yaxis()

# x 轴范围自动留边
all_values = np.concatenate([
    plot_df["agent_OR"].to_numpy(),
    plot_df["observer_OR"].to_numpy(),
    plot_df["human_OR"].to_numpy()
])

x_min = max(0.70, np.nanmin(all_values) - 0.06)
x_max = min(1.70, np.nanmax(all_values) + 0.08)

# 为了接近你附件图的视觉范围，这里固定到 0.75--1.60
ax.set_xlim(0.75, 1.60)
ax.set_xticks(np.arange(0.8, 1.61, 0.1))

ax.tick_params(axis="x", labelsize=30, width=1.6, length=8)
ax.tick_params(axis="y", labelsize=34, width=1.6, length=0)

ax.set_xlabel(
    "Odds ratio (OR)",
    fontsize=40,
    labelpad=26
)

# 网格
ax.grid(
    axis="x",
    color=grid_color,
    linestyle="-",
    linewidth=1.5,
    alpha=0.9
)
ax.grid(axis="y", visible=False)

# 边框
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_linewidth(1.4)
ax.spines["bottom"].set_color("#b0b0b0")

# legend
legend_handles = [
    Line2D(
        [0], [0],
        marker="D",
        color="none",
        markerfacecolor=agent_color,
        markeredgecolor="white",
        markeredgewidth=1.2,
        markersize=20,
        label="LLM (First-person)"
    ),
    Line2D(
        [0], [0],
        marker="s",
        color="none",
        markerfacecolor=observer_color,
        markeredgecolor="white",
        markeredgewidth=1.2,
        markersize=20,
        label="LLM (Observer)"
    ),
    Line2D(
        [0], [0],
        marker="|",
        color=human_color,
        linestyle="None",
        markeredgewidth=4.0,
        markersize=34,
        label="Human Reference"
    ),
]

ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=3,
    frameon=False,
    fontsize=32,
    handletextpad=0.8,
    columnspacing=2.8
)

plt.subplots_adjust(
    left=0.29,
    right=0.985,
    top=0.97,
    bottom=0.24
)

plt.savefig(
    "mechanism_feature_or_dumbbell_large_font.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    "mechanism_feature_or_dumbbell_large_font.pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
