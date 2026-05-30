import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 1. 路径设置
# =========================
STRATEGY_PATH = "/data7/chenyitong/Winning_Arguments/single_turn_pairs_with_strategies.json"

FIRST_PERSON_MODEL_PATHS = {
    "DeepSeek-V3": "/data7/chenyitong/Winning_Arguments/final_v3_results_deepseek-ai_DeepSeek-V3.json",
    "Gemini-2.5-Flash-Lite": "/data7/chenyitong/Winning_Arguments/final_v3_results_google_gemini-2.5-flash-lite.json",
    "GPT-4o-mini": "/data7/chenyitong/Winning_Arguments/final_v3_results_gpt-4o-mini.json",
    "MiniMax-M2.5": "/data7/chenyitong/Winning_Arguments/final_v3_results_Pro_MiniMaxAI_MiniMax-M2.5.json",
    "GLM-4.7": "/data7/chenyitong/Winning_Arguments/final_v3_results_Pro_zai-org_GLM-4.7.json",
    "Qwen-32B": "/data7/chenyitong/Winning_Arguments/final_v3_results_Qwen_Qwen2.5-32B-Instruct.json",
    "Qwen-72B": "/data7/chenyitong/Winning_Arguments/final_v3_results_Qwen_Qwen2.5-72B-Instruct.json",
}

OBSERVER_MODEL_PATHS = {
    "DeepSeek-V3": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_deepseek-ai_DeepSeek-V3.json",
    "Gemini-2.5-Flash-Lite": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_google_gemini-2.5-flash-lite.json",
    "GPT-4o-mini": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_gpt-4o-mini.json",
    "MiniMax-M2.5": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_Pro_MiniMaxAI_MiniMax-M2.5.json",
    "GLM-4.7": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_Pro_zai-org_GLM-4.7.json",
    "Qwen-32B": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_Qwen_Qwen2.5-32B-Instruct.json",
    "Qwen-72B": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_Qwen_Qwen2.5-72B-Instruct.json",
}

OUT_DIR = Path("/data7/chenyitong/Winning_Arguments/strategy_effectiveness_plot_first_vs_observer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2. 工具函数
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_bool_like(x):
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)) and x in [0, 1]:
        return int(x)
    if isinstance(x, float) and x in [0.0, 1.0]:
        return int(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ["true", "1", "yes", "y", "success", "persuasive"]:
            return 1
        if s in ["false", "0", "no", "n", "failure", "not persuasive"]:
            return 0
    return None

def get_prediction_from_branch(branch):
    """
    第一人称文件通常使用 agent_delta；
    第三人称/observer 文件通常使用 delta_awarded。
    """
    for key in ["agent_delta", "delta_awarded", "agent_delta_awarded", "predicted_success", "prediction"]:
        if key in branch:
            val = parse_bool_like(branch.get(key))
            if val is not None:
                return val
    return None

def strategy_combo_label(strategy_dict):
    logos = strategy_dict.get("logos", False)
    pathos = strategy_dict.get("pathos", False)
    ethos = strategy_dict.get("ethos", False)

    active = []
    if logos:
        active.append("L")
    if ethos:
        active.append("E")
    if pathos:
        active.append("P")

    if not active:
        return "None"

    if active == ["L"]:
        return "Logos"
    elif active == ["E"]:
        return "Ethos"
    elif active == ["P"]:
        return "Pathos"
    else:
        return "+".join(active)

def build_strategy_lookup(strategy_data):
    lookup = {}
    for pair_id, pair_obj in strategy_data.items():
        if "success" in pair_obj:
            combo = strategy_combo_label(pair_obj["success"].get("persuasion_strategies", {}))
            lookup[(pair_id, "success")] = combo
        if "failure" in pair_obj:
            combo = strategy_combo_label(pair_obj["failure"].get("persuasion_strategies", {}))
            lookup[(pair_id, "failure")] = combo
    return lookup

def build_reply_level_records(strategy_lookup, model_name, model_results, perspective):
    rows = []

    for item in model_results:
        pair_id = item.get("pair_id")
        if pair_id is None:
            continue

        if "branch_A_human_success" in item and isinstance(item["branch_A_human_success"], dict):
            branch = item["branch_A_human_success"]
            combo = strategy_lookup.get((pair_id, "success"), "None")
            human_label = parse_bool_like(branch.get("human_label"))
            if human_label is None:
                human_label = 1
            pred = get_prediction_from_branch(branch)

            if pred is not None:
                rows.append({
                    "pair_id": pair_id,
                    "reply_branch": "success",
                    "strategy_combo": combo,
                    "human_label": int(human_label),
                    "agent_pred": int(pred),
                    "model": model_name,
                    "perspective": perspective
                })

        if "branch_B_human_failure" in item and isinstance(item["branch_B_human_failure"], dict):
            branch = item["branch_B_human_failure"]
            combo = strategy_lookup.get((pair_id, "failure"), "None")
            human_label = parse_bool_like(branch.get("human_label"))
            if human_label is None:
                human_label = 0
            pred = get_prediction_from_branch(branch)

            if pred is not None:
                rows.append({
                    "pair_id": pair_id,
                    "reply_branch": "failure",
                    "strategy_combo": combo,
                    "human_label": int(human_label),
                    "agent_pred": int(pred),
                    "model": model_name,
                    "perspective": perspective
                })

    return rows

# =========================
# 3. 读取 strategy 数据
# =========================
strategy_data = load_json(STRATEGY_PATH)
strategy_lookup = build_strategy_lookup(strategy_data)

# =========================
# 4. 读取两种视角模型结果并构造长表
# =========================
all_rows = []

for model_name, model_path in FIRST_PERSON_MODEL_PATHS.items():
    results = load_json(model_path)
    rows = build_reply_level_records(strategy_lookup, model_name, results, perspective="First-person")
    all_rows.extend(rows)

for model_name, model_path in OBSERVER_MODEL_PATHS.items():
    results = load_json(model_path)
    rows = build_reply_level_records(strategy_lookup, model_name, results, perspective="Observer")
    all_rows.extend(rows)

df = pd.DataFrame(all_rows)

if df.empty:
    raise RuntimeError("No valid rows were extracted. Please check result-file fields and paths.")

df = df[df["strategy_combo"] != "None"].copy()
df.to_csv(OUT_DIR / "reply_level_strategy_results_long_first_vs_observer.csv", index=False, encoding="utf-8-sig")

# =========================
# 5. 计算 human rate
# =========================
human_df = df[["pair_id", "reply_branch", "strategy_combo", "human_label"]].drop_duplicates()

human_summary = (
    human_df
    .groupby("strategy_combo")
    .agg(
        n=("human_label", "size"),
        human_rate=("human_label", "mean")
    )
    .reset_index()
)

# =========================
# 6. 计算每个模型、每个视角下各策略组合的 persuasion rate
# =========================
model_summary = (
    df
    .groupby(["strategy_combo", "perspective", "model"])
    .agg(
        llm_rate=("agent_pred", "mean")
    )
    .reset_index()
)

plot_df = model_summary.merge(human_summary, on="strategy_combo", how="left")

# =========================
# 7. 计算 across-model 平均
# =========================
mean_df = (
    plot_df
    .groupby(["strategy_combo", "perspective"])
    .agg(
        human_rate=("human_rate", "mean"),
        llm_rate_mean=("llm_rate", "mean"),
        n=("n", "first")
    )
    .reset_index()
)

order_df = human_summary.sort_values("n", ascending=False).reset_index(drop=True)
combo_order = order_df["strategy_combo"].tolist()

plot_df["strategy_combo"] = pd.Categorical(plot_df["strategy_combo"], categories=combo_order, ordered=True)
mean_df["strategy_combo"] = pd.Categorical(mean_df["strategy_combo"], categories=combo_order, ordered=True)
human_summary["strategy_combo"] = pd.Categorical(human_summary["strategy_combo"], categories=combo_order, ordered=True)

human_summary.to_csv(OUT_DIR / "human_strategy_summary.csv", index=False, encoding="utf-8-sig")
plot_df.to_csv(OUT_DIR / "model_strategy_summary_first_vs_observer.csv", index=False, encoding="utf-8-sig")
mean_df.to_csv(OUT_DIR / "mean_strategy_summary_first_vs_observer.csv", index=False, encoding="utf-8-sig")

# =========================
# 8. 颜色
# =========================
color_map = {
    "Logos": "#2E86DE",
    "Pathos": "#8E5B4C",
    "Ethos": "#F1949B",
    "L+P": "#F39C12",
    "L+E": "#27AE60",
    "E+P": "#7D6BC2",
    "L+E+P": "#E74C3C",
}
default_color = "#7f8c8d"

# =========================
# 9A. 保留原始风格散点图：分别保存 Human vs 第一人称 / Human vs Observer
# =========================
def draw_scatter_for_one_perspective(sub_plot_df, sub_mean_df, perspective_name, out_prefix):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)

    all_x = sub_plot_df["human_rate"].tolist() + sub_mean_df["human_rate"].tolist()
    all_y = sub_plot_df["llm_rate"].tolist() + sub_mean_df["llm_rate_mean"].tolist()

    min_val = min(all_x + all_y)
    max_val = max(all_x + all_y)

    pad = 0.08
    low = max(0.0, min_val - pad)
    high = min(1.0, max_val + pad)

    xs = np.linspace(low, high, 400)
    ax.fill_between(xs, xs, high, color="#7aa6d8", alpha=0.08)
    ax.fill_between(xs, low, xs, color="#d99b9b", alpha=0.08)
    ax.plot([low, high], [low, high], linestyle="--", color="gray", linewidth=1.0, alpha=0.7)

    for combo in combo_order:
        sub = sub_plot_df[sub_plot_df["strategy_combo"] == combo]
        color = color_map.get(combo, default_color)
        ax.scatter(
            sub["human_rate"],
            sub["llm_rate"],
            s=36,
            color=color,
            alpha=0.38,
            edgecolor="none",
            zorder=2
        )

    for _, row in sub_mean_df.iterrows():
        combo = row["strategy_combo"]
        color = color_map.get(combo, default_color)
        ax.scatter(
            row["human_rate"],
            row["llm_rate_mean"],
            s=90,
            marker="D",
            color=color,
            edgecolor="white",
            linewidth=1.5,
            zorder=4
        )
        ax.text(
            row["human_rate"] + 0.01,
            row["llm_rate_mean"] + 0.005,
            str(combo),
            fontsize=10,
            color=color,
            weight="bold"
        )

    ax.text(low + 0.01, high - 0.05, "LLM > Human", color="#7aa6d8", fontsize=11, weight="bold")
    ax.text(high - 0.20, low + 0.03, "LLM < Human", color="#d98880", fontsize=11, weight="bold")

    ax.text(
        high - 0.24, low + 0.01,
        "◆ = mean across 7 models\n● = individual model",
        fontsize=8.5,
        color="dimgray",
        ha="left",
        va="bottom"
    )

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Persuasion Rate (Human)", fontsize=12)
    ax.set_ylabel(f"Persuasion Rate ({perspective_name})", fontsize=12)
    ax.set_title(f"Persuasion Effectiveness by Persuasion Strategy ({perspective_name})", fontsize=14, weight="bold")
    ax.grid(alpha=0.20, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{out_prefix}.png", bbox_inches="tight", dpi=300)
    plt.savefig(OUT_DIR / f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)

for perspective_name in ["First-person", "Observer"]:
    sub_plot = plot_df[plot_df["perspective"] == perspective_name].copy()
    sub_mean = mean_df[mean_df["perspective"] == perspective_name].copy()
    draw_scatter_for_one_perspective(
        sub_plot,
        sub_mean,
        perspective_name=perspective_name,
        out_prefix=f"persuasion_strategy_effectiveness_scatter_{perspective_name.lower().replace('-', '_').replace(' ', '_')}"
    )

# =========================
# 9B. 新逻辑：横轴第一人称 rate，纵轴第三人称 Observer rate
#     小点 = 每个模型在该策略组合下的值
#     菱形 = 7个模型平均值
# =========================
pair_df = (
    plot_df
    .pivot_table(
        index=["strategy_combo", "model"],
        columns="perspective",
        values="llm_rate",
        aggfunc="first"
    )
    .reset_index()
)

pair_df = pair_df.dropna(subset=["First-person", "Observer"]).copy()

mean_pair_df = (
    mean_df
    .pivot_table(
        index="strategy_combo",
        columns="perspective",
        values="llm_rate_mean",
        aggfunc="first"
    )
    .reset_index()
)

mean_pair_df = mean_pair_df.dropna(subset=["First-person", "Observer"]).copy()

pair_df.to_csv(OUT_DIR / "strategy_first_person_vs_observer_model_points.csv", index=False, encoding="utf-8-sig")
mean_pair_df.to_csv(OUT_DIR / "strategy_first_person_vs_observer_mean_points.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(8, 8), dpi=180)

all_x = pair_df["First-person"].tolist() + mean_pair_df["First-person"].tolist()
all_y = pair_df["Observer"].tolist() + mean_pair_df["Observer"].tolist()

min_val = min(all_x + all_y)
max_val = max(all_x + all_y)

pad = 0.08
low = max(0.0, min_val - pad)
high = min(1.0, max_val + pad)

xs = np.linspace(low, high, 400)

# y > x: observer rate higher than first-person
ax.fill_between(xs, xs, high, color="#7aa6d8", alpha=0.08)
# y < x: observer rate lower than first-person
ax.fill_between(xs, low, xs, color="#d99b9b", alpha=0.08)

ax.plot([low, high], [low, high], linestyle="--", color="gray", linewidth=1.1, alpha=0.8)

# individual model points
for combo in combo_order:
    sub = pair_df[pair_df["strategy_combo"] == combo]
    if sub.empty:
        continue
    color = color_map.get(combo, default_color)

    ax.scatter(
        sub["First-person"],
        sub["Observer"],
        s=38,
        color=color,
        alpha=0.40,
        edgecolor="none",
        zorder=2
    )

# mean diamonds
label_offsets = {
    "Logos": (0.010, 0.004),
    "Pathos": (0.010, -0.004),
    "Ethos": (0.010, 0.004),
    "L+P": (0.010, 0.004),
    "L+E": (0.010, 0.004),
    "E+P": (0.010, 0.004),
    "L+E+P": (0.010, 0.004),
}

for _, row in mean_pair_df.iterrows():
    combo = row["strategy_combo"]
    color = color_map.get(combo, default_color)

    ax.scatter(
        row["First-person"],
        row["Observer"],
        s=120,
        marker="D",
        color=color,
        edgecolor="white",
        linewidth=1.6,
        zorder=4
    )

    dx, dy = label_offsets.get(str(combo), (0.010, 0.004))
    ax.text(
        row["First-person"] + dx,
        row["Observer"] + dy,
        str(combo),
        fontsize=11,
        color=color,
        weight="bold",
        zorder=5
    )

ax.text(low + 0.01, high - 0.05, "Observer > First-person", color="#7aa6d8", fontsize=11, weight="bold")
ax.text(high - 0.28, low + 0.03, "Observer < First-person", color="#d98880", fontsize=11, weight="bold")

ax.text(
    high - 0.24,
    low + 0.01,
    "◆ = mean across 7 models\n● = individual model",
    fontsize=8.5,
    color="dimgray",
    ha="left",
    va="bottom"
)

ax.set_xlim(low, high)
ax.set_ylim(low, high)
ax.set_xlabel("Persuasion Rate (First-person)", fontsize=12)
ax.set_ylabel("Persuasion Rate (Observer)", fontsize=12)
ax.set_title("Persuasion Effectiveness: First-person vs Observer", fontsize=14, weight="bold")
ax.grid(alpha=0.20, linestyle="--")
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT_DIR / "persuasion_strategy_first_person_vs_observer_scatter.png", bbox_inches="tight", dpi=300)
plt.savefig(OUT_DIR / "persuasion_strategy_first_person_vs_observer_scatter.pdf", bbox_inches="tight")
plt.close(fig)

print(f"Done. Outputs saved to: {OUT_DIR}")
