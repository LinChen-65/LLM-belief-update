import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportions_ztest
from sklearn.metrics import confusion_matrix


STRATEGY_FILE = "/data7/chenyitong/Winning_Arguments/single_turn_pairs_with_strategies.json"

MODEL_FILES = {
    "DeepSeek-V3": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_deepseek-ai_DeepSeek-V3.json",
    "Gemini-2.5-Flash": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_google_gemini-2.5-flash-lite.json",
    "GPT-4o-mini": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_gpt-4o-mini.json",
    "MiniMax-M2.5": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_Pro_MiniMaxAI_MiniMax-M2.5.json",
    "GLM-4.7": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_Pro_zai-org_GLM-4.7.json",
    "Qwen-32B": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_Qwen_Qwen2.5-32B-Instruct.json",
    "Qwen-72B": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_Qwen_Qwen2.5-72B-Instruct.json",
}

OUTPUT_DIR = "/data7/chenyitong/Winning_Arguments/final_v3_Experiment_Analysis_Results/observer_Strategy_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def calculate_kappa_and_se(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    n = np.sum(cm)
    if n == 0:
        return np.nan, np.nan

    p0 = np.sum(np.diag(cm)) / n
    p_e = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (n ** 2)

    if p_e == 1:
        return 1.0, 0.0

    kappa = (p0 - p_e) / (1 - p_e)
    se_kappa = np.sqrt((p0 * (1 - p0)) / (n * (1 - p_e) ** 2)) if n > 0 else np.nan
    return float(kappa), float(se_kappa)


def compare_kappas(k1, se1, k2, se2):
    if pd.isna(k1) or pd.isna(k2) or pd.isna(se1) or pd.isna(se2):
        return np.nan
    if se1 == 0 and se2 == 0:
        return 1.0
    z_score = abs(k1 - k2) / np.sqrt(se1 ** 2 + se2 ** 2)
    return float(2 * (1 - norm.cdf(z_score)))


def compare_proportions(count1, nobs1, count2, nobs2):
    if nobs1 == 0 or nobs2 == 0:
        return np.nan
    counts = np.array([count1, count2])
    nobs = np.array([nobs1, nobs2])
    if min(counts) < 0 or min(nobs - counts) < 0:
        return np.nan
    _, p_value = proportions_ztest(counts, nobs)
    return float(p_value)


def compare_paired_binary_mcnemar(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if len(y_true) == 0:
        return np.nan

    table = confusion_matrix(y_true, y_pred, labels=[0, 1])
    try:
        res = mcnemar(table, exact=False, correction=True)
        return float(res.pvalue)
    except Exception:
        return np.nan


def get_star(p):
    if pd.isna(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def draw_significance_bars(ax, x1, x2, y, h, text):
    line_color = "#555555"
    ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h], lw=1.2, c=line_color)

    if text == "ns":
        text_color = "#888888"
        font_weight = "normal"
        font_size = 10
    else:
        text_color = "black"
        font_weight = "bold"
        font_size = 13

    ax.text(
        (x1 + x2) / 2,
        y + (h * 0.2),
        text,
        ha="center",
        va="bottom",
        color=text_color,
        fontsize=font_size,
        fontweight=font_weight,
    )


def sanitize_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def normalize_strategy_combo(strategy_dict):
    if not isinstance(strategy_dict, dict):
        return "None"

    labels = []
    for key in ["logos", "ethos", "pathos"]:
        if strategy_dict.get(key) is True:
            labels.append(key.capitalize())

    if not labels:
        return "None"

    return "+".join(labels)


def branch_specs():
    return [
        ("branch_A_human_success", "success", 1),
        ("branch_B_human_failure", "failure", 0),
        ("success", "success", 1),
        ("failure", "failure", 0),
    ]


def load_strategy_reply_rows():
    print("📦 正在加载论证策略标注数据...")
    if not os.path.exists(STRATEGY_FILE):
        raise FileNotFoundError(f"找不到策略文件: {STRATEGY_FILE}")

    with open(STRATEGY_FILE, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    rows = []
    excluded_none = 0

    for pair_id, pair in pairs.items():
        if not isinstance(pair, dict):
            continue

        for branch_key, condition, human_label in branch_specs():
            branch = pair.get(branch_key)
            if not isinstance(branch, dict):
                continue

            strategies = branch.get("persuasion_strategies", {})
            combo = normalize_strategy_combo(strategies)

            if combo == "None":
                excluded_none += 1
                continue

            rows.append({
                "pair_id": pair_id,
                "root_id": branch.get("root") or branch.get("reply-to"),
                "reply_id": branch.get("id"),
                "condition": condition,
                "Human_Label": human_label,
                "Strategy_Combo": combo,
                "logos": bool(strategies.get("logos", False)),
                "ethos": bool(strategies.get("ethos", False)),
                "pathos": bool(strategies.get("pathos", False)),
            })

    df = pd.DataFrame(rows)

    print(f"  ✅ 策略标注 reply rows: {len(df)}")
    print(f"  ⚠️ 三种策略均为 false/缺失而被排除的案例数: {excluded_none}")

    return df, excluded_none


def parse_bool_like(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "yes", "1"}:
            return True
        if s in {"false", "no", "0"}:
            return False
        m = re.search(r'"?(?:agent_delta|delta_awarded|predicted_success)"?\s*[:=]\s*(true|false)', s)
        if m:
            return m.group(1) == "true"
        try:
            parsed = json.loads(value)
            return parse_agent_bool_from_obj(parsed)
        except Exception:
            return None
    return None


def parse_agent_bool_from_obj(obj):
    """
    Extract the model judgment from a branch object.

    New observer-result structure:
        branch_A_human_success / branch_B_human_failure:
            human_label: 1/0
            delta_awarded: true/false
            justification: ...

    The script keeps backward compatibility with older result files that used
    agent_delta, agent_delta_awarded, predicted_success, or prediction.
    """
    if isinstance(obj, bool):
        return obj

    if isinstance(obj, dict):
        for key in [
            "delta_awarded",
            "agent_delta",
            "agent_delta_awarded",
            "predicted_success",
            "prediction",
            "pred",
            "answer",
            "decision",
            "is_persuasive",
            "persuaded",
        ]:
            if key in obj:
                b = parse_bool_like(obj.get(key))
                if b is not None:
                    return b

        for key in ["response", "content", "message", "output", "raw_output", "text", "completion", "result"]:
            if key in obj:
                b = parse_bool_like(obj.get(key))
                if b is not None:
                    return b

    if isinstance(obj, str):
        return parse_bool_like(obj)

    return None


def extract_agent_predictions_for_model(filepath, model_name):
    """
    Parse model result files into:
        Model, pair_id, condition, Agent_Label

    Compatible with the new branch-format observer structure:
        {
            "pair_id": "p_4263",
            "root_id": "t3_35bc4b",
            "evaluation_mode": "observer",
            "branch_A_human_success": {
                "human_label": 1,
                "delta_awarded": false,
                "justification": "..."
            },
            "branch_B_human_failure": {
                "human_label": 0,
                "delta_awarded": false,
                "justification": "..."
            }
        }

    The matching key remains (pair_id, condition), so the downstream merge with
    strategy_df is unchanged.
    """
    if not os.path.exists(filepath):
        print(f"  ⚠️ 模型文件不存在，跳过: {model_name} -> {filepath}")
        return pd.DataFrame(columns=["Model", "pair_id", "condition", "Agent_Label"])

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ["data", "results", "items", "records"]:
            if key in data and isinstance(data[key], list):
                data_iter = data[key]
                break
        else:
            data_iter = list(data.values())
    else:
        data_iter = data

    records = []
    skipped_non_observer = 0
    skipped_missing_pair = 0
    skipped_missing_label = 0

    for entry in data_iter:
        if not isinstance(entry, dict):
            continue

        evaluation_mode = str(entry.get("evaluation_mode", "")).strip().lower()
        if evaluation_mode and evaluation_mode != "observer":
            skipped_non_observer += 1
            continue

        pair_id = entry.get("pair_id") or entry.get("pair") or entry.get("pair_key")
        if pair_id is None:
            skipped_missing_pair += 1
            continue

        branch_map = [
            ("branch_A_human_success", "success"),
            ("branch_B_human_failure", "failure"),
            ("branch_A_success", "success"),
            ("branch_B_failure", "failure"),
            ("success", "success"),
            ("failure", "failure"),
        ]

        for branch_key, condition in branch_map:
            branch = entry.get(branch_key)
            if not isinstance(branch, dict):
                continue

            label = parse_agent_bool_from_obj(branch)
            if label is None:
                skipped_missing_label += 1
                continue

            records.append({
                "Model": model_name,
                "pair_id": str(pair_id),
                "condition": condition,
                "Agent_Label": 1 if label else 0,
                "prediction_field": (
                    "delta_awarded" if "delta_awarded" in branch
                    else "agent_delta" if "agent_delta" in branch
                    else "agent_delta_awarded" if "agent_delta_awarded" in branch
                    else "other"
                ),
                "evaluation_mode": evaluation_mode or "unknown",
            })

    df = pd.DataFrame(records)

    if not df.empty:
        df = df.drop_duplicates(subset=["Model", "pair_id", "condition"], keep="first")

    print(
        f"  ✅ {model_name}: extracted predictions = {len(df)} "
        f"(non_observer_skipped={skipped_non_observer}, "
        f"missing_pair={skipped_missing_pair}, missing_label={skipped_missing_label})"
    )

    return df
def load_all_data():
    strategy_df, excluded_none = load_strategy_reply_rows()

    all_model_rows = []
    parse_reports = []

    for model_name, filepath in MODEL_FILES.items():
        pred_df = extract_agent_predictions_for_model(filepath, model_name)

        if pred_df.empty:
            parse_reports.append({
                "Model": model_name,
                "File": filepath,
                "Extracted_Predictions": 0,
                "Matched_Rows": 0,
                "Unmatched_Rows": len(strategy_df),
            })
            continue

        merged = strategy_df.merge(
            pred_df,
            on=["pair_id", "condition"],
            how="left",
            validate="one_to_one",
        )

        matched = merged.dropna(subset=["Agent_Label"]).copy()
        matched["Agent_Label"] = matched["Agent_Label"].astype(int)
        matched["Model"] = model_name

        all_model_rows.append(matched)

        parse_reports.append({
            "Model": model_name,
            "File": filepath,
            "Extracted_Predictions": len(pred_df),
            "Matched_Rows": len(matched),
            "Unmatched_Rows": int(merged["Agent_Label"].isna().sum()),
            "Prediction_Fields": ";".join(sorted(pred_df["prediction_field"].dropna().unique())) if "prediction_field" in pred_df.columns else "",
        })

    if all_model_rows:
        df = pd.concat(all_model_rows, ignore_index=True)
    else:
        df = pd.DataFrame()

    report_df = pd.DataFrame(parse_reports)

    return strategy_df, df, report_df, excluded_none


def compute_distribution(strategy_df, excluded_none):
    total_included = len(strategy_df)
    dist = (
        strategy_df
        .groupby("Strategy_Combo")
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
    dist["Proportion"] = dist["Count"] / total_included if total_included > 0 else np.nan
    dist["Excluded_None_Count"] = excluded_none
    return dist


def compute_human_success_rates(strategy_df):
    human = (
        strategy_df
        .groupby("Strategy_Combo")
        .agg(
            Human_Success_Rate=("Human_Label", "mean"),
            Count=("Human_Label", "size"),
            Success_Count=("Human_Label", "sum"),
        )
        .reset_index()
        .sort_values(["Human_Success_Rate", "Count"], ascending=[False, False])
    )
    return human


def compute_model_success_rates(df):
    rows = []

    for model, df_m in df.groupby("Model"):
        for combo, sub in df_m.groupby("Strategy_Combo"):
            human_rate = sub["Human_Label"].mean()
            agent_rate = sub["Agent_Label"].mean()
            p_mcnemar = compare_paired_binary_mcnemar(sub["Human_Label"], sub["Agent_Label"])

            rows.append({
                "Model": model,
                "Strategy_Combo": combo,
                "N": len(sub),
                "Human_Success_Rate": human_rate,
                "Agent_Success_Rate": agent_rate,
                "Difference_Agent_minus_Human": agent_rate - human_rate,
                "McNemar_p": p_mcnemar,
                "McNemar_sig": get_star(p_mcnemar),
            })

    return pd.DataFrame(rows)


def compute_alignment_by_combo(df):
    rows = []

    for model, df_m in df.groupby("Model"):
        for combo, sub in df_m.groupby("Strategy_Combo"):
            y_true = sub["Human_Label"].astype(int).values
            y_pred = sub["Agent_Label"].astype(int).values

            if len(sub) == 0:
                continue

            kappa, se_k = calculate_kappa_and_se(y_true, y_pred)
            accuracy = float(np.mean(y_true == y_pred))
            fp_count = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn_count = int(np.sum((y_true == 1) & (y_pred == 0)))
            human_pos = int(np.sum(y_true == 1))
            human_neg = int(np.sum(y_true == 0))
            agent_pos = int(np.sum(y_pred == 1))

            rows.append({
                "Model": model,
                "Strategy_Combo": combo,
                "N": len(sub),
                "Cohen_Kappa": kappa,
                "SE_Kappa": se_k,
                "Accuracy": accuracy,
                "FP_Count": fp_count,
                "FN_Count": fn_count,
                "Human_Positive_Count": human_pos,
                "Human_Negative_Count": human_neg,
                "Agent_Positive_Count": agent_pos,
                "FP_Rate": fp_count / human_neg if human_neg > 0 else np.nan,
                "FN_Rate": fn_count / human_pos if human_pos > 0 else np.nan,
            })

    return pd.DataFrame(rows)



def normalize_combo_color_key(combo):
    """
    Normalize strategy-combination names so that names from plot_strategy_distribution.py
    such as "Ethos + Logos" and names from this script such as "Logos+Ethos"
    receive the same color.
    """
    if combo is None or pd.isna(combo):
        return "None"

    parts = [p.strip() for p in str(combo).replace(" + ", "+").split("+") if p.strip()]
    if not parts:
        return "None"

    canonical_order = {"Ethos": 0, "Logos": 1, "Pathos": 2}
    parts = [p.capitalize() for p in parts]
    parts = sorted(parts, key=lambda x: canonical_order.get(x, 99))
    return "+".join(parts)


def build_strategy_color_map(dist_df):
    """
    Reproduce the color-selection logic in plot_strategy_distribution.py:
    sorted strategies by Count descending, then assign colors from matplotlib Set3.
    The resulting mapping is reused by every strategy-level plot in this script.
    """
    df_color = dist_df.copy()
    if "Strategy_Combination" in df_color.columns and "Strategy_Combo" not in df_color.columns:
        df_color = df_color.rename(columns={"Strategy_Combination": "Strategy_Combo"})

    if "Count" in df_color.columns:
        df_color = df_color.sort_values(by="Count", ascending=False)

    cmap = plt.get_cmap("Set3")
    color_map = {}

    for i, combo in enumerate(df_color["Strategy_Combo"].tolist()):
        key = normalize_combo_color_key(combo)
        if key not in color_map:
            color_map[key] = cmap(i % 12)

    return color_map


def get_strategy_colors(combo_list, strategy_color_map):
    fallback = plt.get_cmap("Set3")
    colors = []
    for i, combo in enumerate(combo_list):
        key = normalize_combo_color_key(combo)
        colors.append(strategy_color_map.get(key, fallback(i % 12)))
    return colors


def save_strategy_color_mapping(strategy_color_map):
    from matplotlib.colors import to_hex

    rows = []
    for combo, color in strategy_color_map.items():
        rows.append({
            "Strategy_Combo_Normalized": combo,
            "Color_Hex": to_hex(color),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "strategy_color_mapping_set3.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def plot_strategy_distribution(dist_df, strategy_color_map):
    order = dist_df["Strategy_Combo"].tolist()
    colors = get_strategy_colors(order, strategy_color_map)

    plt.figure(figsize=(14, 8))
    ax = sns.barplot(
        data=dist_df,
        x="Strategy_Combo",
        y="Proportion",
        order=order,
        palette=colors,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, color="gray")
    plt.title("Distribution of Persuasion Strategy Combinations", fontsize=18, fontweight="bold", pad=18)
    plt.xlabel("Strategy Combination", fontsize=14)
    plt.ylabel("Proportion of Included Replies", fontsize=14)
    plt.xticks(rotation=35, ha="right", fontsize=12)
    plt.yticks(fontsize=12)

    ymax = max(dist_df["Proportion"].max() * 1.22, 0.1)
    ax.set_ylim(0, ymax)

    for p, (_, row) in zip(ax.patches, dist_df.iterrows()):
        ax.text(
            p.get_x() + p.get_width() / 2,
            p.get_height() + ymax * 0.015,
            f"{int(row['Count'])}\n({row['Proportion']:.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333333",
        )

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "Strategy_Distribution_Count_Proportion.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 已保存: {save_path}")


def plot_human_success_rates(human_df, strategy_color_map):
    order = human_df["Strategy_Combo"].tolist()
    colors = get_strategy_colors(order, strategy_color_map)

    plt.figure(figsize=(14, 8))
    ax = sns.barplot(
        data=human_df,
        x="Strategy_Combo",
        y="Human_Success_Rate",
        order=order,
        palette=colors,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, color="gray")
    plt.title("Human Persuasion Success Rate by Strategy Combination", fontsize=18, fontweight="bold", pad=18)
    plt.xlabel("Strategy Combination", fontsize=14)
    plt.ylabel("Human Success Rate", fontsize=14)
    plt.xticks(rotation=35, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_ylim(0, min(1.15, max(1.0, human_df["Human_Success_Rate"].max() * 1.18)))

    for p, (_, row) in zip(ax.patches, human_df.iterrows()):
        ax.text(
            p.get_x() + p.get_width() / 2,
            p.get_height() + 0.015,
            f"{row['Human_Success_Rate']:.1%}\n(n={int(row['Count'])})",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333333",
        )

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "Human_Success_Rate_by_Strategy.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 已保存: {save_path}")


def plot_model_vs_human_success_rates(model_rate_df, human_order, strategy_color_map):
    """
    Bars are colored by Strategy_Combo to match plot_strategy_distribution.py.
    Human and Agent bars are distinguished by transparency/hatch rather than by strategy color.
    """
    from matplotlib.patches import Patch

    for model, sub in model_rate_df.groupby("Model"):
        sub = sub.copy()
        sub["Strategy_Combo"] = pd.Categorical(sub["Strategy_Combo"], categories=human_order, ordered=True)
        sub = sub.sort_values("Strategy_Combo")

        x = np.arange(len(human_order))
        width = 0.36
        colors = get_strategy_colors(human_order, strategy_color_map)
        sub_indexed = sub.set_index("Strategy_Combo")

        human_vals = [sub_indexed.loc[c, "Human_Success_Rate"] if c in sub_indexed.index else np.nan for c in human_order]
        agent_vals = [sub_indexed.loc[c, "Agent_Success_Rate"] if c in sub_indexed.index else np.nan for c in human_order]

        fig, ax = plt.subplots(figsize=(16, 8))

        bars_h = ax.bar(
            x - width / 2,
            human_vals,
            width,
            color=colors,
            alpha=0.50,
            edgecolor="black",
            linewidth=0.8,
            hatch="//",
            label="Human",
        )
        bars_a = ax.bar(
            x + width / 2,
            agent_vals,
            width,
            color=colors,
            alpha=0.95,
            edgecolor="black",
            linewidth=0.8,
            label=model,
        )

        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6, color="gray")
        ax.set_title(f"Human vs {model}: Success Rate by Strategy Combination", fontsize=18, fontweight="bold", pad=18)
        ax.set_xlabel("Strategy Combination", fontsize=14)
        ax.set_ylabel("Success Rate", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(human_order, rotation=35, ha="right", fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylim(0, 1.28)

        for bar in list(bars_h) + list(bars_a):
            h = bar.get_height()
            if not pd.isna(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.012,
                    f"{h:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#333333",
                )

        for i, combo in enumerate(human_order):
            if combo not in sub_indexed.index:
                continue

            star = sub_indexed.loc[combo, "McNemar_sig"]
            p_h = bars_h[i]
            p_a = bars_a[i]
            x_h = p_h.get_x() + p_h.get_width() / 2
            x_a = p_a.get_x() + p_a.get_width() / 2
            y = max(p_h.get_height(), p_a.get_height()) + 0.10
            draw_significance_bars(ax, x_h, x_a, y, 0.018, star)

        legend_handles = [
            Patch(facecolor="white", edgecolor="black", hatch="//", label="Human"),
            Patch(facecolor="white", edgecolor="black", label=model),
        ]
        ax.legend(handles=legend_handles, title="Rater", title_fontsize=13, fontsize=12, loc="upper right")

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"Model_vs_Human_Success_Rate_{sanitize_filename(model)}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✅ 已保存: {save_path}")


def plot_model_alignment_by_strategy(alignment_df, combo_order, strategy_color_map):
    for model, sub_all in alignment_df.groupby("Model"):
        sub = sub_all.copy()
        sub["Strategy_Combo"] = pd.Categorical(sub["Strategy_Combo"], categories=combo_order, ordered=True)
        sub = sub.sort_values("Strategy_Combo")
        colors = get_strategy_colors(combo_order, strategy_color_map)

        plt.figure(figsize=(15, 8))
        ax = sns.barplot(
            data=sub,
            x="Strategy_Combo",
            y="Cohen_Kappa",
            order=combo_order,
            palette=colors,
            edgecolor="black",
            linewidth=0.8,
        )

        ax.axhline(0, color="red", linestyle="--", linewidth=1.2, label="Random Chance (Kappa=0)")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6, color="gray")
        plt.title(f"{model}: Alignment with Human by Strategy Combination", fontsize=18, fontweight="bold", pad=18)
        plt.xlabel("Strategy Combination", fontsize=14)
        plt.ylabel("Cohen's Kappa", fontsize=14)
        plt.xticks(rotation=35, ha="right", fontsize=12)
        plt.yticks(fontsize=12)

        kmin = min(0, sub["Cohen_Kappa"].min(skipna=True))
        kmax = max(0.05, sub["Cohen_Kappa"].max(skipna=True))
        lower = kmin - 0.08
        upper = kmax + 0.12
        ax.set_ylim(lower, upper)

        for p, (_, row) in zip(ax.patches, sub.iterrows()):
            h = p.get_height()
            if not pd.isna(h):
                y_text = h + (upper - lower) * 0.025 if h >= 0 else h - (upper - lower) * 0.045
                va = "bottom" if h >= 0 else "top"
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    y_text,
                    f"{h:.3f}\n(n={int(row['N'])})",
                    ha="center",
                    va=va,
                    fontsize=10,
                    color="#333333",
                )

        plt.legend(fontsize=11)
        plt.tight_layout()

        save_path = os.path.join(OUTPUT_DIR, f"Model_Human_Alignment_Kappa_{sanitize_filename(model)}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✅ 已保存: {save_path}")


if __name__ == "__main__":
    strategy_df, df, parse_report_df, excluded_none = load_all_data()

    if strategy_df.empty:
        raise RuntimeError("No strategy-labeled rows were loaded. Check STRATEGY_FILE.")

    parse_report_path = os.path.join(OUTPUT_DIR, "strategy_agent_parse_report.csv")
    parse_report_df.to_csv(parse_report_path, index=False)
    print(f"  ✅ 解析报告已保存至: {parse_report_path}")

    strategy_df.to_csv(os.path.join(OUTPUT_DIR, "strategy_reply_level_human_dataset.csv"), index=False)

    dist_df = compute_distribution(strategy_df, excluded_none)
    dist_df.to_csv(os.path.join(OUTPUT_DIR, "strategy_distribution.csv"), index=False)

    strategy_color_map = build_strategy_color_map(dist_df)
    save_strategy_color_mapping(strategy_color_map)

    human_rate_df = compute_human_success_rates(strategy_df)
    human_rate_df.to_csv(os.path.join(OUTPUT_DIR, "human_success_rate_by_strategy.csv"), index=False)

    if df.empty:
        raise RuntimeError("No model predictions matched the strategy-labeled data. Check model JSON structure and parse report.")

    df.to_csv(os.path.join(OUTPUT_DIR, "strategy_reply_level_agent_human_dataset.csv"), index=False)

    model_rate_df = compute_model_success_rates(df)
    model_rate_df.to_csv(os.path.join(OUTPUT_DIR, "model_vs_human_success_rate_by_strategy.csv"), index=False)

    alignment_df = compute_alignment_by_combo(df)
    alignment_df.to_csv(os.path.join(OUTPUT_DIR, "model_human_alignment_by_strategy.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "strategy_analysis_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "strategy_file": STRATEGY_FILE,
            "model_files": MODEL_FILES,
            "output_dir": OUTPUT_DIR,
            "excluded_none_strategy_count": int(excluded_none),
            "note": "Rows with no true strategy among logos/ethos/pathos are excluded from analysis. Model result files are parsed from the branch-format observer structure; delta_awarded is used as the primary model judgment field, with backward compatibility for agent_delta and related fields.",
            "human_agent_rate_test": "McNemar paired test per strategy combination.",
            "alignment_metric": "Cohen's Kappa between Human_Label and Agent_Label per strategy combination.",
            "strategy_color_rule": "Colors follow plot_strategy_distribution.py: strategy combinations are sorted by Count descending and assigned colors from matplotlib Set3. The same Strategy_Combo uses the same color across plots.",
        }, f, ensure_ascii=False, indent=2)

    print("\n🎨 正在绘制图表...")
    plot_strategy_distribution(dist_df, strategy_color_map)
    plot_human_success_rates(human_rate_df, strategy_color_map)

    human_order = human_rate_df["Strategy_Combo"].tolist()
    plot_model_vs_human_success_rates(model_rate_df, human_order, strategy_color_map)

    for model in alignment_df["Model"].unique():
        order = (
            alignment_df[alignment_df["Model"] == model]
            .sort_values(["Cohen_Kappa", "N"], ascending=[False, False])["Strategy_Combo"]
            .tolist()
        )
        plot_model_alignment_by_strategy(alignment_df[alignment_df["Model"] == model], order, strategy_color_map)

    print("\n🎉 所有任务执行完毕！")
    print(f"结果目录: {OUTPUT_DIR}")
