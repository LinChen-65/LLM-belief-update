import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from statsmodels.stats.proportion import proportions_ztest

# ==========================================
# 1. 配置文件路径
# ==========================================
TOPIC_FILE = '/data7/chenyitong/Winning_Arguments/topic_classification_results.json'

MODEL_FILES = {
    "DeepSeek-V3": "/data7/chenyitong/Winning_Arguments/final_v3_results_deepseek-ai_DeepSeek-V3.json",
    "Gemini-2.5-Flash": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_google_gemini-2.5-flash-lite.json",
    "GPT-4o-mini": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_gpt-4o-mini.json",
    "MiniMax-M2.5": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_Pro_MiniMaxAI_MiniMax-M2.5.json",
    "GLM-4.7": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_Pro_zai-org_GLM-4.7.json",
    "Qwen-32B": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_Qwen_Qwen2.5-32B-Instruct.json",
    "Qwen-72B": "/data7/chenyitong/Winning_Arguments/final_v3_results_observer_Qwen_Qwen2.5-72B-Instruct.json",
}

OUTPUT_DIR = "/data7/chenyitong/Winning_Arguments/final_v3_Experiment_Analysis_Results/observer_Topic_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 核心统计学函数与显著性检验
# ==========================================
def calculate_kappa_and_se(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return 0.0, 0.0
    n = np.sum(cm)
    if n == 0:
        return 0.0, 0.0
    p0 = np.sum(np.diag(cm)) / n
    p_e = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (n ** 2)
    if p_e == 1:
        return 1.0, 0.0
    kappa = (p0 - p_e) / (1 - p_e)
    se_kappa = np.sqrt((p0 * (1 - p0)) / (n * (1 - p_e) ** 2))
    return kappa, se_kappa


def compare_kappas(k1, se1, k2, se2):
    if se1 == 0 and se2 == 0:
        return 1.0
    denom = np.sqrt(se1 ** 2 + se2 ** 2)
    if denom == 0:
        return 1.0
    z_score = abs(k1 - k2) / denom
    p_value = 2 * (1 - norm.cdf(z_score))
    return p_value


def compare_proportions(count1, nobs1, count2, nobs2):
    if nobs1 == 0 or nobs2 == 0:
        return 1.0
    counts = np.array([count1, count2])
    nobs = np.array([nobs1, nobs2])
    if min(counts) < 0 or min(nobs - counts) < 0:
        return 1.0
    stat, p_value = proportions_ztest(counts, nobs)
    return p_value


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

# ==========================================
# 3. 数据加载与清洗
# ==========================================
def parse_binary_label(value):
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int) and value in [0, 1]:
        return int(value)
    if isinstance(value, float) and value in [0.0, 1.0]:
        return int(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ["true", "1", "yes", "y"]:
            return 1
        if v in ["false", "0", "no", "n"]:
            return 0
    return None


def get_model_prediction(branch):
    """
    兼容新旧结构。
    新 observer 结构优先读取 delta_awarded：
        "delta_awarded": false
    旧结构保留兼容：
        "agent_delta": false
        "agent_delta_awarded": false
    """
    for key in ["delta_awarded", "agent_delta", "agent_delta_awarded", "predicted_delta", "prediction"]:
        if key in branch:
            parsed = parse_binary_label(branch.get(key))
            if parsed is not None:
                return parsed
    return None


def get_human_label(branch, default_label=None):
    if "human_label" in branch:
        parsed = parse_binary_label(branch.get("human_label"))
        if parsed is not None:
            return parsed
    return default_label


def normalize_topic(x):
    if x is None:
        return None
    x = str(x).strip().lower()
    if x in ["fact", "value", "policy"]:
        return x.capitalize()
    return None


def unwrap_json_list(data, filepath):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "results", "items", "records"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"{filepath} 的JSON结构不是list，也没有可识别的list字段。")


def load_data():
    print("📦 正在加载话题分类数据...")
    if not os.path.exists(TOPIC_FILE):
        raise FileNotFoundError(f"找不到Topic文件: {TOPIC_FILE}")

    with open(TOPIC_FILE, 'r', encoding='utf-8') as f:
        topics_data = json.load(f)

    topic_map = {}
    for item in topics_data:
        root_id = item.get("id") or item.get("root_id")
        topic = normalize_topic(item.get("proposition_type"))
        if root_id and topic:
            topic_map[root_id] = topic

    records = []
    skipped_files = []
    skipped_no_topic = 0
    skipped_no_prediction = 0

    for model_name, filepath in MODEL_FILES.items():
        if not os.path.exists(filepath):
            print(f"⚠️ 文件不存在，跳过: {model_name} -> {filepath}")
            skipped_files.append((model_name, filepath))
            continue

        print(f"  正在读取: {model_name}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = unwrap_json_list(json.load(f), filepath)

        for item in data:
            root_id = item.get('root_id')
            topic = topic_map.get(root_id)
            if topic not in ['Fact', 'Value', 'Policy']:
                skipped_no_topic += 1
                continue

            evaluation_mode = str(item.get("evaluation_mode", "")).strip().lower()
            if evaluation_mode and evaluation_mode != "observer":
                continue

            branch_specs = [
                ('branch_A_human_success', 1),
                ('branch_B_human_failure', 0)
            ]

            for branch_key, default_human_label in branch_specs:
                branch = item.get(branch_key, {})
                if not isinstance(branch, dict):
                    skipped_no_prediction += 1
                    continue

                human_label = get_human_label(branch, default_label=default_human_label)
                agent_label = get_model_prediction(branch)

                if human_label is None or agent_label is None:
                    skipped_no_prediction += 1
                    continue

                records.append({
                    'Model': model_name,
                    'Topic': topic,
                    'Human_Label': int(human_label),
                    'Agent_Label': int(agent_label),
                    'Pair_ID': item.get('pair_id'),
                    'Root_ID': root_id,
                    'Branch': branch_key,
                    'Evaluation_Mode': item.get('evaluation_mode', 'observer')
                })

    if skipped_files:
        skipped_path = os.path.join(OUTPUT_DIR, "skipped_missing_files.csv")
        pd.DataFrame(skipped_files, columns=["Model", "Filepath"]).to_csv(skipped_path, index=False)
        print(f"⚠️ 缺失文件列表已保存: {skipped_path}")

    df = pd.DataFrame(records)

    if df.empty:
        print("❌ 没有读取到任何有效记录。请检查 MODEL_FILES 路径、delta_awarded 字段和 root_id-topic 匹配。")
    else:
        print(f"✅ 成功读取 {len(df)} 条 reply-level 记录。")
        print("Topic分布：")
        print(df['Topic'].value_counts().to_string())
        print("Model分布：")
        print(df['Model'].value_counts().to_string())
        print(f"跳过无topic匹配记录数: {skipped_no_topic}")
        print(f"跳过无有效预测分支数: {skipped_no_prediction}")

    return df

# ==========================================
# 4. 统计与计算显著性连线
# ==========================================
def compute_stats_and_pvalues(df):
    topics = ['Fact', 'Value', 'Policy']
    models = df['Model'].unique().tolist()
    results = []

    for model in models:
        df_m = df[df['Model'] == model]
        model_stats = {}

        for topic in topics:
            df_t = df_m[df_m['Topic'] == topic]
            if df_t.empty:
                model_stats[topic] = None
                continue

            y_true = df_t['Human_Label'].values
            y_pred = df_t['Agent_Label'].values
            kappa, se_k = calculate_kappa_and_se(y_true, y_pred)

            df_h1 = df_t[df_t['Human_Label'] == 1]
            df_h0 = df_t[df_t['Human_Label'] == 0]

            fn_count = len(df_h1[df_h1['Agent_Label'] == 0])
            fn_nobs = len(df_h1)
            fp_count = len(df_h0[df_h0['Agent_Label'] == 1])
            fp_nobs = len(df_h0)

            fn_rate = fn_count / fn_nobs if fn_nobs > 0 else 0
            fp_rate = fp_count / fp_nobs if fp_nobs > 0 else 0

            model_stats[topic] = {
                'Kappa': kappa,
                'SE_Kappa': se_k,
                'FN_Count': fn_count,
                'FN_Nobs': fn_nobs,
                'FP_Count': fp_count,
                'FP_Nobs': fp_nobs,
                'FN_Rate': fn_rate,
                'FP_Rate': fp_rate
            }

            results.append({
                'Model': model,
                'Topic': topic,
                'N': len(df_t),
                'Cohen_Kappa': kappa,
                'FN_Rate': fn_rate,
                'FP_Rate': fp_rate,
                'FN_Count': fn_count,
                'FN_Nobs': fn_nobs,
                'FP_Count': fp_count,
                'FP_Nobs': fp_nobs
            })

        pairs = [('Fact', 'Value'), ('Value', 'Policy'), ('Fact', 'Policy')]
        for metric, func in [('Cohen_Kappa', 'kappa'), ('FN_Rate', 'prop_fn'), ('FP_Rate', 'prop_fp')]:
            for t1, t2 in pairs:
                p_val = 1.0
                if model_stats.get(t1) and model_stats.get(t2):
                    if func == 'kappa':
                        p_val = compare_kappas(
                            model_stats[t1]['Kappa'], model_stats[t1]['SE_Kappa'],
                            model_stats[t2]['Kappa'], model_stats[t2]['SE_Kappa']
                        )
                    elif func == 'prop_fn':
                        p_val = compare_proportions(
                            model_stats[t1]['FN_Count'], model_stats[t1]['FN_Nobs'],
                            model_stats[t2]['FN_Count'], model_stats[t2]['FN_Nobs']
                        )
                    elif func == 'prop_fp':
                        p_val = compare_proportions(
                            model_stats[t1]['FP_Count'], model_stats[t1]['FP_Nobs'],
                            model_stats[t2]['FP_Count'], model_stats[t2]['FP_Nobs']
                        )

                for r in results:
                    if r['Model'] == model:
                        r[f'p_{metric}_{t1}_vs_{t2}'] = p_val

    return pd.DataFrame(results)

# ==========================================
# 5. 辅助绘图函数
# ==========================================
def draw_significance_bars(ax, x1, x2, y, h, text):
    line_color = '#555555'
    ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h], lw=1.2, c=line_color)

    if text == "ns":
        text_color = "#888888"
        font_weight = "normal"
        font_size = 11
    else:
        text_color = "black"
        font_weight = "bold"
        font_size = 15

    ax.text((x1 + x2) / 2, y + (h * 0.2), text,
            ha='center', va='bottom', color=text_color,
            fontsize=font_size, fontweight=font_weight)

# ==========================================
# 6. 绘图 1: 分模型小图排布大图
# ==========================================
def plot_grid_view_for_metric(res_df, metric_col, title):
    models = res_df['Model'].unique()
    num_models = len(models)
    cols = 4
    rows = (num_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows))
    axes = axes.flatten()

    topics = ['Fact', 'Value', 'Policy']
    colors = sns.color_palette("Set2", 3)

    for i, model in enumerate(models):
        ax = axes[i]
        df_m = res_df[res_df['Model'] == model].set_index('Topic')

        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='gray')

        x_pos = np.arange(len(topics))
        vals = [df_m.loc[t, metric_col] if t in df_m.index else 0 for t in topics]

        ax.bar(x_pos, vals, color=colors, width=0.6, edgecolor='black', linewidth=0.8)

        ax.set_title(model, fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(topics, fontsize=14)
        ax.tick_params(axis='y', labelsize=12)

        local_max = max(vals) if max(vals) > 0 else 0.1
        ylim_max = local_max * 2.2
        ax.set_ylim(0, ylim_max)

        p_fv = get_star(df_m.iloc[0].get(f'p_{metric_col}_Fact_vs_Value', 1.0))
        p_vp = get_star(df_m.iloc[0].get(f'p_{metric_col}_Value_vs_Policy', 1.0))
        p_fp = get_star(df_m.iloc[0].get(f'p_{metric_col}_Fact_vs_Policy', 1.0))

        h_tick = ylim_max * 0.02
        v_step = ylim_max * 0.12

        y1 = max(vals[0], vals[1]) + v_step
        draw_significance_bars(ax, 0, 1, y1, h_tick, p_fv)

        y2 = max(vals[1], vals[2]) + v_step
        y2 = max(y2, y1 + v_step)
        draw_significance_bars(ax, 1, 2, y2, h_tick, p_vp)

        y3 = max(vals) + v_step
        y3 = max(y3, y2 + v_step)
        draw_significance_bars(ax, 0, 2, y3, h_tick, p_fp)

        for j, val in enumerate(vals):
            if val > 0:
                ax.text(j, val + (ylim_max * 0.015), f"{val:.3f}",
                        ha='center', va='bottom', fontsize=11, color='black')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'{title} Across Models', fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"GridView_{metric_col}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 已保存: {save_path}")

# ==========================================
# 7. 绘图 2: 全局对比视图
# ==========================================
def plot_global_view_with_sig(res_df, metric_col, title):
    fact_data = res_df[res_df['Topic'] == 'Fact'][['Model', metric_col]].set_index('Model')
    sorted_models = fact_data.sort_values(by=metric_col, ascending=False).index.tolist()

    plt.figure(figsize=(24, 10))
    ax = sns.barplot(data=res_df, x='Model', y=metric_col, hue='Topic', order=sorted_models,
                     palette='Set2', edgecolor='black', linewidth=0.8)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='gray')

    plt.title(f'{title} by Model (Sorted by Fact)', fontsize=22, fontweight='bold', pad=20)
    plt.ylabel(title, fontsize=16)
    plt.xlabel('Model', fontsize=16)
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14)

    global_max = res_df[metric_col].max()
    ylim_max = global_max * 2.2 if global_max > 0 else 1
    ax.set_ylim(0, ylim_max)

    bars = ax.patches
    num_models = len(sorted_models)

    h_tick = ylim_max * 0.015
    v_step = ylim_max * 0.1

    for p in bars:
        h = p.get_height()
        if not pd.isna(h) and h > 0:
            ax.text(p.get_x() + p.get_width() / 2, h + (ylim_max * 0.01), f"{h:.3f}",
                    ha='center', va='bottom', fontsize=10, rotation=45, color='#333333')

    for i, model in enumerate(sorted_models):
        b_fact = bars[i]
        b_value = bars[i + num_models]
        b_policy = bars[i + 2 * num_models]

        x_f, y_f = b_fact.get_x() + b_fact.get_width() / 2, b_fact.get_height()
        x_v, y_v = b_value.get_x() + b_value.get_width() / 2, b_value.get_height()
        x_p, y_p = b_policy.get_x() + b_policy.get_width() / 2, b_policy.get_height()

        df_m = res_df[res_df['Model'] == model].iloc[0]

        p_fv = get_star(df_m.get(f'p_{metric_col}_Fact_vs_Value', 1.0))
        p_vp = get_star(df_m.get(f'p_{metric_col}_Value_vs_Policy', 1.0))
        p_fp = get_star(df_m.get(f'p_{metric_col}_Fact_vs_Policy', 1.0))

        y_max_fv = max(y_f, y_v) + v_step * 1.5
        draw_significance_bars(ax, x_f, x_v, y_max_fv, h_tick, p_fv)

        y_max_vp = max(y_v, y_p) + v_step * 1.5
        y_max_vp = max(y_max_vp, y_max_fv + v_step)
        draw_significance_bars(ax, x_v, x_p, y_max_vp, h_tick, p_vp)

        y_max_fp = max([y_f, y_v, y_p]) + v_step * 1.5
        y_max_fp = max(y_max_fp, y_max_vp + v_step)
        draw_significance_bars(ax, x_f, x_p, y_max_fp, h_tick, p_fp)

    plt.legend(title='Proposition Type', title_fontsize=14, fontsize=13,
               bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"GlobalView_{metric_col}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 已保存: {save_path}")

# ==========================================
# 8. 主执行流程
# ==========================================
if __name__ == "__main__":
    df = load_data()

    if df is not None and not df.empty:
        raw_path = os.path.join(OUTPUT_DIR, "observer_topic_reply_level_records.csv")
        df.to_csv(raw_path, index=False)
        print(f"  ✅ reply-level原始记录已保存至: {raw_path}")

        print("\n📊 正在计算指标与跨 Topic 的显著性检验...")
        res_df = compute_stats_and_pvalues(df)

        csv_path = os.path.join(OUTPUT_DIR, "topic_analysis_results_with_pvalues.csv")
        res_df.to_csv(csv_path, index=False)
        print(f"  ✅ 计算结果已保存至: {csv_path}")

        print("\n🎨 正在绘制可视化图表...")
        metrics = [
            ('Cohen_Kappa', "Cohen's Kappa Score"),
            ('FN_Rate', "FN Rate (Stubbornness)"),
            ('FP_Rate', "FP Rate (Yielding)")
        ]

        for m_col, title in metrics:
            plot_grid_view_for_metric(res_df, m_col, title)
            plot_global_view_with_sig(res_df, m_col, title)

        print("\n🎉 所有任务执行完毕！")
