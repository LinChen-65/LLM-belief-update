

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import cohen_kappa_score


# ==========================================
# 1. 基础配置：observer 文件路径
# ==========================================
OBSERVER_MODEL_PATHS = {
    "DeepSeek-V3": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_deepseek-ai_DeepSeek-V3.json",
    "Gemini-2.5-Flash-Lite": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_google_gemini-2.5-flash-lite.json",
    "GPT-4o-mini": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_gpt-4o-mini.json",
    "MiniMax-M2.5": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_Pro_MiniMaxAI_MiniMax-M2.5.json",
    "GLM-4.7": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_Pro_zai-org_GLM-4.7.json",
    "Qwen-32B": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_Qwen_Qwen2.5-32B-Instruct.json",
    "Qwen-72B": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_Qwen_Qwen2.5-72B-Instruct.json",
}

OUTPUT_DIR = "/data7/chenyitong/Winning_Arguments/final_v3_Experiment_Analysis_Results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "Observer_FP_FN_Kappa_Plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ==========================================
# 2. 工具函数
# ==========================================
def parse_bool_label(x):
    if isinstance(x, bool):
        return int(x)

    if isinstance(x, int) and x in [0, 1]:
        return int(x)

    if isinstance(x, float) and x in [0.0, 1.0]:
        return int(x)

    if isinstance(x, str):
        s = x.strip().lower()
        if s in ["true", "1", "yes", "y"]:
            return 1
        if s in ["false", "0", "no", "n"]:
            return 0

    return None


def get_observer_prediction(branch_data):
    """
    observer 文件的预测字段是 delta_awarded。
    为兼容旧格式，同时保留 agent_delta。
    """
    for key in ["delta_awarded", "agent_delta", "agent_delta_awarded", "prediction", "predicted_success"]:
        if key in branch_data:
            value = parse_bool_label(branch_data.get(key))
            if value is not None:
                return value

    return None


# ==========================================
# 3. 数据处理与展平
# ==========================================
def load_and_flatten_data(model_name, file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在，跳过: {file_path}")
        return pd.DataFrame()

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ JSON 解析失败，跳过: {file_path}")
            return pd.DataFrame()

    records = []

    for entry in data:
        pair_id = entry.get("pair_id")
        root_id = entry.get("root_id")
        evaluation_mode = entry.get("evaluation_mode", "observer")

        branch_specs = [
            ("branch_A_human_success", 1),
            ("branch_B_human_failure", 0),
        ]

        for branch_name, default_human_label in branch_specs:
            branch_data = entry.get(branch_name, {})

            if not isinstance(branch_data, dict):
                continue

            human_label = parse_bool_label(
                branch_data.get("human_label", default_human_label)
            )
            if human_label is None:
                human_label = default_human_label

            observer_pred = get_observer_prediction(branch_data)

            if observer_pred is None:
                continue

            if human_label == 1 and observer_pred == 1:
                matrix_type = "TP"
            elif human_label == 1 and observer_pred == 0:
                matrix_type = "FN"
            elif human_label == 0 and observer_pred == 0:
                matrix_type = "TN"
            elif human_label == 0 and observer_pred == 1:
                matrix_type = "FP"
            else:
                matrix_type = "Unknown"

            records.append({
                "Model": model_name,
                "Pair_ID": pair_id,
                "Root_ID": root_id,
                "Evaluation_Mode": evaluation_mode,
                "Branch": branch_name,
                "Human_Label": human_label,
                "Observer_Pred": observer_pred,
                "Matrix_Type": matrix_type,
            })

    return pd.DataFrame(records)


# ==========================================
# 4. 核心分析与绘图
# ==========================================
def run_final_analysis():
    df_list = []

    for model_name, file_path in OBSERVER_MODEL_PATHS.items():
        model_df = load_and_flatten_data(model_name, file_path)
        if not model_df.empty:
            df_list.append(model_df)

    if not df_list:
        print("❌ 没有成功读取任何 observer 模型文件。请检查路径。")
        return

    all_df = pd.concat(df_list, ignore_index=True)

    if all_df.empty:
        print("❌ 所有文件读取后为空。")
        return

    clean_df = all_df[all_df["Matrix_Type"] != "Unknown"].copy()

    model_stats = {}

    for model in list(OBSERVER_MODEL_PATHS.keys()):
        m_df = all_df[all_df["Model"] == model]

        if m_df.empty:
            print(f"⚠️ {model} 没有有效数据，跳过。")
            continue

        fp_count = len(m_df[m_df["Matrix_Type"] == "FP"])
        fn_count = len(m_df[m_df["Matrix_Type"] == "FN"])

        m_clean = clean_df[clean_df["Model"] == model]

        y_true = m_clean["Human_Label"].tolist()
        y_pred = m_clean["Observer_Pred"].tolist()

        kappa = cohen_kappa_score(y_true, y_pred) if len(y_true) > 0 else 0

        # 如果你想严格保持原代码逻辑，可以继续固定为 6032
        TOTAL_SAMPLES = 6032

        fp_pct = (fp_count / TOTAL_SAMPLES) * 100
        fn_pct = (fn_count / TOTAL_SAMPLES) * 100

        model_stats[model] = {
            "FP_Count": fp_count,
            "FN_Count": fn_count,
            "N_Valid": len(m_clean),
            "FP_Pct": fp_pct,
            "FN_Pct": fn_pct,
            "Total_F": fp_pct + fn_pct,
            "Kappa": kappa,
        }

    if not model_stats:
        print("❌ 没有可用于绘图的统计结果。")
        return

    # 保存展平数据
    flat_output_path = os.path.join(
        OUTPUT_DIR,
        "observer_flattened_confusion_data.csv"
    )
    all_df.to_csv(flat_output_path, index=False, encoding="utf-8-sig")

    # 保存模型统计表
    stats_df = pd.DataFrame.from_dict(model_stats, orient="index").reset_index()
    stats_df = stats_df.rename(columns={"index": "Model"})

    stats_output_path = os.path.join(
        OUTPUT_DIR,
        "observer_fp_fn_kappa_summary.csv"
    )
    stats_df.to_csv(stats_output_path, index=False, encoding="utf-8-sig")

    # ==========================
    # 5. 绘图
    # ==========================
    bar_width = 0.55

    sorted_models = sorted(
        model_stats.keys(),
        key=lambda x: model_stats[x]["Total_F"],
        reverse=True
    )

    fn_pcts = [model_stats[m]["FN_Pct"] for m in sorted_models]
    fp_pcts = [model_stats[m]["FP_Pct"] for m in sorted_models]
    kappas = [model_stats[m]["Kappa"] for m in sorted_models]

    COLOR_FN = "#2C3E50"
    COLOR_FP = "#E74C3C"
    COLOR_KAPPA = "#27AE60"

    fig, ax1 = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    ax1.bar(
        sorted_models,
        fn_pcts,
        width=bar_width,
        color=COLOR_FN,
        label="FN"
    )

    ax1.bar(
        sorted_models,
        fp_pcts,
        width=bar_width,
        bottom=fn_pcts,
        color=COLOR_FP,
        label="FP"
    )

    ax1.set_ylabel(
        "Percentage of Total Cases (%)",
        fontsize=12,
        fontweight="bold"
    )

    ax1.set_xticks(range(len(sorted_models)))
    ax1.set_xticklabels(
        sorted_models,
        rotation=35,
        ha="right",
        fontsize=11
    )

    max_y1 = max([f + p for f, p in zip(fn_pcts, fp_pcts)])
    ax1.set_ylim(0, max_y1 * 1.15)

    ax2 = ax1.twinx()

    ax2.plot(
        sorted_models,
        kappas,
        color=COLOR_KAPPA,
        marker="o",
        markersize=9,
        linewidth=3,
        markeredgecolor="white",
        markeredgewidth=1.5,
        label="Kappa"
    )

    ax2.set_ylabel(
        "Cohen's Kappa Score",
        fontsize=12,
        fontweight="bold",
        color=COLOR_KAPPA
    )

    ax2.tick_params(axis="y", colors=COLOR_KAPPA)
    ax2.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.5
    )

    min_kappa = min(kappas)
    max_kappa = max(kappas)

    ax2.set_ylim(
        min(0, min_kappa) - 0.02,
        max(0.1, max_kappa) + 0.05
    )

    ax1.set_zorder(1)
    ax2.set_zorder(2)
    ax2.patch.set_visible(False)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        fontsize=12,
        frameon=False
    )

    plt.title(
        "Observer Anomalies (FP & FN) and Kappa Across All Models",
        fontsize=16,
        fontweight="bold",
        y=1.12
    )

    bbox_props = dict(
        boxstyle="round,pad=0.2",
        facecolor="white",
        alpha=0.7,
        edgecolor="none"
    )

    for i in range(len(sorted_models)):
        fn_val = fn_pcts[i]
        fp_val = fp_pcts[i]
        total_val = fn_val + fp_val
        k_val = kappas[i]

        if fn_val > 0:
            ax1.text(
                i,
                fn_val / 2,
                f"{fn_val:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=10
            )

        if fp_val > 0:
            ax1.text(
                i,
                fn_val + (fp_val * 0.75),
                f"{fp_val:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=10
            )

        ax1.text(
            i,
            total_val + (max_y1 * 0.04),
            f"Total: {total_val:.1f}%",
            ha="center",
            va="bottom",
            color="black",
            fontweight="bold",
            fontsize=11
        )

        y_offset = 0.005 if k_val >= 0 else -0.015
        va_val = "bottom" if k_val >= 0 else "top"

        ax2.text(
            i,
            k_val + y_offset,
            f"{k_val:.3f}",
            ha="center",
            va=va_val,
            color=COLOR_KAPPA,
            fontweight="black",
            fontsize=12,
            bbox=bbox_props
        )

    plt.tight_layout()

    output_path = os.path.join(
        PLOTS_DIR,
        "Observer_All_Models_FP_FN_Kappa.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        facecolor="white",
        bbox_inches="tight"
    )
    plt.close()
    
    # ==========================
    # 6. 生成 TXT 摘要文件
    # ==========================
    txt_output_path = os.path.join(OUTPUT_DIR, "Observer_All_Models_Metrics_Data.txt")
    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write("Observer 模型评估数据摘要 (按 Total FP+FN 降序排列)\n")
        f.write("=" * 60 + "\n")
        for model in sorted_models:
            stats = model_stats[model]
            f.write(f"模型名称: {model}\n")
            f.write(f"  - FN 占比:    {stats['FN_Pct']:.2f}%\n")
            f.write(f"  - FP 占比:    {stats['FP_Pct']:.2f}%\n")
            f.write(f"  - 总异常占比: {stats['Total_F']:.2f}%\n")
            f.write(f"  - Kappa 得分: {stats['Kappa']:.4f}\n")
            f.write("-" * 60 + "\n")

    print("✅ Observer 分析完成")
    print(f"展平数据已保存: {flat_output_path}")
    print(f"模型统计已保存: {stats_output_path}")
    print(f"图像已保存: {output_path}")
    print(f"TXT统计已保存: {txt_output_path}")


if __name__ == "__main__":
    run_final_analysis()
