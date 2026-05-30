import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import cohen_kappa_score

# ==========================================
# 1. 基础配置
# ==========================================
FILES = {
    "Qwen-32B": "final_v3_observer_results_Qwen_Qwen2.5-32B-Instruct.json",
    "Qwen-72B": "final_v3_observer_results_Qwen_Qwen2.5-72B-Instruct.json",
    "DeepSeek-V3": "final_v3_observer_results_deepseek-ai_DeepSeek-V3.json",
    "MiniMax-M2.5": "final_v3_observer_results_Pro_MiniMaxAI_MiniMax-M2.5.json",
    "Gemini-2.5-Flash-Lite": "final_v3_observer_results_google_gemini-2.5-flash-lite.json",
    "GLM-4.7": "final_v3_observer_results_Pro_zai-org_GLM-4.7.json",
    "gpt-4o-mini": "final_v3_observer_results_openai_gpt-4o-mini.json"
}

OUTPUT_DIR = "final_v3_Experiment_Analysis_Results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "Plots_Images")
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 数据处理与展平
# ==========================================
def load_and_flatten_data(model_name, file_path):
    if not os.path.exists(file_path): return pd.DataFrame()
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError: return pd.DataFrame()
        
    records = []
    for entry in data:
        for branch_name, human_label in [('branch_A_human_success', 1), ('branch_B_human_failure', 0)]:
            branch_data = entry.get(branch_name, {})
            agent_delta = branch_data.get('agent_delta')
            
            if agent_delta not in [True, False]: continue
            
            if human_label == 1 and agent_delta == True: matrix_type = 'TP'
            elif human_label == 1 and agent_delta == False: matrix_type = 'FN'
            elif human_label == 0 and agent_delta == False: matrix_type = 'TN'
            elif human_label == 0 and agent_delta == True: matrix_type = 'FP'
            else: matrix_type = 'Unknown'
                
            records.append({
                'Model': model_name,
                'Human_Label': human_label,
                'Agent_Delta': 1 if agent_delta else 0,
                'Matrix_Type': matrix_type
            })
    return pd.DataFrame(records)

# ==========================================
# 3. 核心绘图与数据保存逻辑
# ==========================================
def run_final_analysis():
    df_list = [load_and_flatten_data(m, p) for m, p in FILES.items()]
    all_df = pd.concat(df_list, ignore_index=True)
    if all_df.empty: return

    clean_df = all_df[all_df['Matrix_Type'] != 'Unknown']
    model_stats = {}
    
    for model in list(FILES.keys()):
        m_df = all_df[all_df['Model'] == model]
        if m_df.empty: continue
        
        fp_count = len(m_df[m_df['Matrix_Type'] == 'FP'])
        fn_count = len(m_df[m_df['Matrix_Type'] == 'FN'])
        
        m_clean = clean_df[clean_df['Model'] == model]
        y_true, y_pred = m_clean['Human_Label'].tolist(), m_clean['Agent_Delta'].tolist()
        
        k = cohen_kappa_score(y_true, y_pred) if len(y_true) > 0 else 0
        
        # 统一使用 6032 作为分母
        TOTAL_SAMPLES = 6032 
        fp_pct, fn_pct = (fp_count / TOTAL_SAMPLES) * 100, (fn_count / TOTAL_SAMPLES) * 100
        
        model_stats[model] = {'FP_Pct': fp_pct, 'FN_Pct': fn_pct, 'Total_F': fp_pct + fn_pct, 'Kappa': k}

    # ================= 绘图部分 =================
    bar_width = 0.55
    sorted_models = sorted(model_stats.keys(), key=lambda x: model_stats[x]['Total_F'], reverse=True)
    fn_pcts = [model_stats[m]['FN_Pct'] for m in sorted_models]
    fp_pcts = [model_stats[m]['FP_Pct'] for m in sorted_models]
    kappas = [model_stats[m]['Kappa'] for m in sorted_models]
    
    # 颜色配置 (高对比度)
    COLOR_FN, COLOR_FP, COLOR_KAPPA = '#2C3E50', '#E74C3C', '#27AE60'
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # 绘制左轴的堆叠柱状图
    bar1 = ax1.bar(sorted_models, fn_pcts, width=bar_width, color=COLOR_FN, label='FN')
    bar2 = ax1.bar(sorted_models, fp_pcts, width=bar_width, bottom=fn_pcts, color=COLOR_FP, label='FP')
    
    ax1.set_ylabel('Percentage of Total Cases (%)', fontsize=12, fontweight='bold') 
    ax1.set_xticks(range(len(sorted_models)))
    ax1.set_xticklabels(sorted_models, rotation=35, ha='right', fontsize=11)
    
    # 恢复自然的左轴比例，仅留出 15% 的头部空间
    max_y1 = max([f + p for f, p in zip(fn_pcts, fp_pcts)])
    ax1.set_ylim(0, max_y1 * 1.15) 
    
    # 创建右轴
    ax2 = ax1.twinx()
    line1 = ax2.plot(sorted_models, kappas, color=COLOR_KAPPA, marker='o', markersize=9, 
                     linewidth=3, markeredgecolor='white', markeredgewidth=1.5, label="Kappa")
    
    ax2.set_ylabel("Cohen's Kappa Score", fontsize=12, fontweight='bold', color=COLOR_KAPPA)
    ax2.tick_params(axis='y', colors=COLOR_KAPPA)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # 恢复自然的右轴比例
    min_kappa, max_kappa = min(kappas), max(kappas)
    ax2.set_ylim(min(0, min_kappa) - 0.02, max(0.1, max_kappa) + 0.05)

    # =============== 核心防遮挡层级设置 ===============
    ax1.set_zorder(1)
    ax2.set_zorder(2)
    ax2.patch.set_visible(False) 

    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', 
               bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize=12, frameon=False)
    
    plt.title('Anomalies (FP & FN) and Kappa Across All Models', fontsize=16, fontweight='bold', y=1.12)
    
    # =============== 标注数值 (带半透明底垫 & 位置微调) ===============
    bbox_props = dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none")
    
    for i in range(len(sorted_models)):
        fn_val, fp_val, total_val, k_val = fn_pcts[i], fp_pcts[i], fn_pcts[i] + fp_pcts[i], kappas[i]
        
        # FN 标注在正中间
        if fn_val > 0:
            ax1.text(i, fn_val / 2, f"{fn_val:.1f}%", ha='center', va='center', color='white', 
                     fontweight='bold', fontsize=10)
        
        # 🌟 修改点 1：将 FP 比例文字往上移，放在 FP 红色柱子高度的 75% 处（避开绿色的线）
        if fp_val > 0:
            ax1.text(i, fn_val + (fp_val * 0.75), f"{fp_val:.1f}%", ha='center', va='center', color='white', 
                     fontweight='bold', fontsize=10)
            
        # 🌟 修改点 2：增加 Total 文字离柱子顶部的间距（乘以 0.04 而非 0.01），悬空更高
        ax1.text(i, total_val + (max_y1 * 0.04), f'Total: {total_val:.1f}%', ha='center', va='bottom', 
                 color='black', fontweight='bold', fontsize=11)
        
        # Kappa 标注：带有白色半透明背景垫
        y_offset = 0.005 if k_val >= 0 else -0.015
        va_val = 'bottom' if k_val >= 0 else 'top'
        ax2.text(i, k_val + y_offset, f"{k_val:.3f}", ha='center', va=va_val, 
                 color=COLOR_KAPPA, fontweight='black', fontsize=12, bbox=bbox_props)

    plt.tight_layout()
    img_output_path = os.path.join(PLOTS_DIR, "All_Models_DualAxis_NaturalScale_FixedText.png")
    plt.savefig(img_output_path, dpi=300)
    plt.close()
    print(f"图表已生成: {img_output_path}")

    # ================= 写入TXT文件部分 =================
    txt_output_path = os.path.join(OUTPUT_DIR, "All_Models_Metrics_Data.txt")
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        f.write("模型评估数据摘要 (按 Total FP+FN 降序排列)\n")
        f.write("=" * 60 + "\n")
        
        for model in sorted_models:
            stats = model_stats[model]
            f.write(f"模型名称: {model}\n")
            f.write(f"  - FN 占比:    {stats['FN_Pct']:.2f}%\n")
            f.write(f"  - FP 占比:    {stats['FP_Pct']:.2f}%\n")
            f.write(f"  - 总异常占比: {stats['Total_F']:.2f}%\n")
            f.write(f"  - Kappa 得分: {stats['Kappa']:.4f}\n")
            f.write("-" * 60 + "\n")
            
    print(f"数据已保存: {txt_output_path}")

if __name__ == "__main__":
    run_final_analysis()
