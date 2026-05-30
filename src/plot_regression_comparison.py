import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 确保在无界面的服务器上不会崩溃
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ==========================================
# 1. 配置文件路径
# ==========================================
AGENT_FILE = '/data7/chenyitong/Winning_Arguments/final_new_Experiment_Analysis_Results/agent_regression/Regression_Report_POOLED_All_Models.txt'
HUMAN_FILE = '/data7/chenyitong/Winning_Arguments/final_new_Experiment_Analysis_Results/Human_Regression/Logistic_Regression_Coefficients.txt'

# 输出图片保存路径
OUTPUT_DIR = os.path.dirname(os.path.dirname(HUMAN_FILE))
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, 'Human_vs_Agent_OR_Comparison_Sorted.png')

# 设置全局学术绘图字体
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 【核心修复】解析原始 statsmodels 表格
# ==========================================
def parse_regression_txt(file_path, group_name):
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return pd.DataFrame()
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # 寻找标准的 statsmodels 原始表格头
    start_idx = -1
    for i, line in enumerate(lines):
        if 'coef' in line and 'std err' in line and 'P>|z|' in line:
            start_idx = i + 2  # 跳过下划线行
            break
            
    if start_idx == -1:
        print(f"❌ 未在 {file_path} 中找到特征汇总表！")
        return pd.DataFrame()
        
    data = []
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if not line or '=====' in line:
            if not line: continue
            else: break
            
        parts = line.split()
        if len(parts) >= 7:
            feature_raw = parts[0]
            # 过滤掉截距和 C(Model) 固定效应变量
            if feature_raw.startswith('Intercept') or feature_raw.startswith('C('):
                continue
                
            feature = feature_raw.replace('_', ' ')
            coef = float(parts[1])
            p_val = float(parts[4])
            
            # 提取 95% 置信区间的 coef 并去掉方括号
            ci_low_coef = float(parts[5].replace('[', '').replace(']', ''))
            ci_high_coef = float(parts[6].replace('[', '').replace(']', ''))
            
            # 使用 exp 转换回 OR 视角
            or_val = np.exp(coef)
            ci_low = np.exp(ci_low_coef)
            ci_high = np.exp(ci_high_coef)
            
            # 重新打上显著性星号
            if p_val < 0.001: sig = '***'
            elif p_val < 0.01: sig = '**'
            elif p_val < 0.05: sig = '*'
            else: sig = ''
            
            data.append({
                'Feature': feature,
                'Coef': coef,
                'OR': or_val,
                'CI_Lower': ci_low,
                'CI_Upper': ci_high,
                'Sig': sig,
                'Group': group_name
            })
            
    return pd.DataFrame(data)

# ==========================================
# 3. 统计检验: Z-test 比较两个独立回归系数
# ==========================================
def compare_same_direction_coefficients(df_human, df_agent):
    print("\n" + "="*80)
    print(" 📊 统计显著性检验：呈现【同向】回归系数的大小差异 (Z-Test for Independent Coefficients)")
    print("="*80)
    
    # 将两个表按特征合并
    df_merged = pd.merge(
        df_human[['Feature', 'Coef', 'CI_Lower', 'CI_Upper']],
        df_agent[['Feature', 'Coef', 'CI_Lower', 'CI_Upper']],
        on='Feature', suffixes=('_Human', '_Agent')
    )
    
    results = []
    
    for _, row in df_merged.iterrows():
        coef_h = row['Coef_Human']
        coef_a = row['Coef_Agent']
        
        if coef_h * coef_a > 0:
            # 逆推 Standard Error (利用 1.96 * 2 = 3.92)
            se_h = (np.log(row['CI_Upper_Human']) - np.log(row['CI_Lower_Human'])) / 3.92
            se_a = (np.log(row['CI_Upper_Agent']) - np.log(row['CI_Lower_Agent'])) / 3.92
            
            z_stat = (coef_a - coef_h) / np.sqrt(se_h**2 + se_a**2)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            if p_value < 0.05:
                if abs(coef_a) > abs(coef_h):
                    conclusion = "Agent 效应显著更强 (Agent > Human)"
                else:
                    conclusion = "Human 效应显著更强 (Human > Agent)"
            else:
                conclusion = "无显著差异 (Agent ≈ Human)"
                
            results.append({
                'Feature': row['Feature'],
                'β(Human)': round(coef_h, 4),
                'β(Agent)': round(coef_a, 4),
                'Z-value': round(z_stat, 3),
                'P-value': f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}",
                'Conclusion': conclusion
            })
            
    if results:
        res_df = pd.DataFrame(results)
        print(res_df.to_string(index=False))
    else:
        print("未发现呈现同向作用的特征。")
    print("="*80 + "\n")

# ==========================================
# 4. 绘制对比森林图 (Forest Plot) - 已按 Human OR 值排序
# ==========================================
def plot_comparison(df_human, df_agent):
    print(f"🎨 正在绘制 Human vs Agent 回归 OR 值对比图 (已按 Human OR 从小到大排序)...")
    
    df_human_sorted = df_human.sort_values(by='OR', ascending=True)
    features = df_human_sorted['Feature'].tolist()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(features))
    offset = 0.15 
    colors = {'Human': '#3498db', 'Agent (LLM)': '#e74c3c'}
    
    # 绘制 1.0 无效假设基准线
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5, zorder=1)
    
    for i, feature in enumerate(features):
        human_row = df_human[df_human['Feature'] == feature]
        agent_row = df_agent[df_agent['Feature'] == feature]
        
        # 绘制人类 (蓝色圆点)
        if not human_row.empty:
            h_or = human_row.iloc[0]['OR']
            h_ci_l = human_row.iloc[0]['CI_Lower']
            h_ci_u = human_row.iloc[0]['CI_Upper']
            h_sig = human_row.iloc[0]['Sig']
            
            ax.plot([h_ci_l, h_ci_u], [i + offset, i + offset], color=colors['Human'], linewidth=2.5, zorder=2)
            ax.scatter(h_or, i + offset, color=colors['Human'], s=80, label='Human' if i==len(features)-1 else "", zorder=3)
            if h_sig:
                ax.text(h_ci_u + 0.02, i + offset, h_sig, color=colors['Human'], va='center', fontweight='bold', fontsize=12)

        # 绘制 Agent (红色菱形)
        if not agent_row.empty:
            a_or = agent_row.iloc[0]['OR']
            a_ci_l = agent_row.iloc[0]['CI_Lower']
            a_ci_u = agent_row.iloc[0]['CI_Upper']
            a_sig = agent_row.iloc[0]['Sig']
            
            ax.plot([a_ci_l, a_ci_u], [i - offset, i - offset], color=colors['Agent (LLM)'], linewidth=2.5, zorder=2)
            ax.scatter(a_or, i - offset, color=colors['Agent (LLM)'], s=80, marker='D', label='Agent (LLM)' if i==len(features)-1 else "", zorder=3)
            if a_sig:
                ax.text(a_ci_u + 0.02, i - offset, a_sig, color=colors['Agent (LLM)'], va='center', fontweight='bold', fontsize=12)

    # 图表修饰
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Odds Ratio (OR)', fontsize=13, fontweight='bold')
    ax.set_title('Persuasion Mechanisms: Human vs. Agent (Sorted by Human OR)', fontsize=15, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    
    ax.text(0.98, -1.0, '← Negative Effect', ha='right', va='center', color='gray', fontsize=11, fontstyle='italic')
    ax.text(1.02, -1.0, 'Positive Effect →', ha='left', va='center', color='gray', fontsize=11, fontstyle='italic')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎉 任务完成！已根据 Human OR 值重新排序，对比图已保存至: {OUTPUT_IMAGE}")

# ==========================================
# 主运行逻辑
# ==========================================
if __name__ == "__main__":
    df_agent = parse_regression_txt(AGENT_FILE, 'Agent (LLM)')
    df_human = parse_regression_txt(HUMAN_FILE, 'Human')
    
    if not df_agent.empty and not df_human.empty:
        compare_same_direction_coefficients(df_human, df_agent)
        plot_comparison(df_human, df_agent)
    else:
        print("❌ 数据解析失败，请检查文件路径或内容格式。")
