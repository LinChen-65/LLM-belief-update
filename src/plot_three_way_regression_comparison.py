import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 确保在无界面的服务器上不会崩溃
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ==========================================
# 1. 配置文件路径
# ==========================================
HUMAN_FILE = '/data7/chenyitong/Winning_Arguments/final_Experiment_Analysis_Results/Human_Regression/Logistic_Regression_Coefficients.txt'
AGENT_1ST_FILE = '/data7/chenyitong/Winning_Arguments/final_Experiment_Analysis_Results/Agent_Regression/Agent_Pooled_Logistic_Regression.txt'
AGENT_3RD_FILE = '/data7/chenyitong/Winning_Arguments/final_Experiment_Analysis_Results/third_Agent_Regression/third_Agent_Pooled_Logistic_Regression.txt'

# 输出图片保存路径（保存在分析结果根目录）
OUTPUT_DIR = '/data7/chenyitong/Winning_Arguments/final_Experiment_Analysis_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, 'Human_vs_Agent1st_vs_Agent3rd_OR_Comparison.png')

# 设置全局学术绘图字体
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 解析 TXT 报告中的回归表格
# ==========================================
def parse_regression_txt(file_path, group_name):
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return pd.DataFrame()
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    data = []
    start_parsing = False
    
    for line in lines:
        # 定位表格表头
        if 'Raw Coef' in line and 'Odds Ratio' in line:
            start_parsing = True
            continue
            
        if start_parsing:
            # 遇到分隔符或空行，说明表格结束
            if '=====' in line or '----' in line or line.strip() == '':
                if len(data) > 0:
                    break
                continue
                
            parts = line.split()
            # 正常的一行数据至少有 6 列 (Feature, Coef, OR, P, CI_L, CI_U)，Sig 可能为空
            if len(parts) >= 6:
                feature = parts[0]
                
                # 过滤掉 Intercept 和 控制变量 C(Model)
                if feature == 'Intercept' or feature.startswith('C(Model)'):
                    continue
                    
                try:
                    or_val = float(parts[2])
                    ci_l = float(parts[4])
                    ci_u = float(parts[5])
                    sig = parts[6] if len(parts) == 7 else ''
                    
                    data.append({
                        'Feature': feature,
                        f'OR_{group_name}': or_val,
                        f'CI_L_{group_name}': ci_l,
                        f'CI_U_{group_name}': ci_u,
                        f'Sig_{group_name}': sig
                    })
                except ValueError:
                    continue
                    
    return pd.DataFrame(data)

# ==========================================
# 3. 数据加载与合并
# ==========================================
print("📦 正在解析回归结果文件...")
df_human = parse_regression_txt(HUMAN_FILE, 'Human')
df_agent_1st = parse_regression_txt(AGENT_1ST_FILE, 'Agent1st')
df_agent_3rd = parse_regression_txt(AGENT_3RD_FILE, 'Agent3rd')

if df_human.empty or df_agent_1st.empty or df_agent_3rd.empty:
    print("❌ 部分文件解析失败或为空，请检查文件路径及格式。")
    exit()

# 将三组数据按照特征 (Feature) 进行内连接交集
df_merged = pd.merge(df_human, df_agent_1st, on='Feature', how='inner')
df_merged = pd.merge(df_merged, df_agent_3rd, on='Feature', how='inner')

# 根据人类的 Odds Ratio 进行升序排序，这样画出来的图会非常整齐
df_merged = df_merged.sort_values(by='OR_Human', ascending=True).reset_index(drop=True)

# ==========================================
# 4. 绘制三元对比森林图 (Forest Plot)
# ==========================================
print("📊 正在绘制三方对比图...")

features = df_merged['Feature']
y_pos = np.arange(len(features))

# 为3个数据点设置不同的纵向偏移量，避免重叠
offset_h = 0.25   # 人类在上方
offset_a1 = 0.0   # Agent 第一人称在中间
offset_a3 = -0.25 # Agent 第三人称在下方

# 定义颜色策略 (绿色：人类，红色：第一人称Agent，蓝色：第三人称Agent)
colors = {
    'Human': '#2ecc71',
    'Agent 1st': '#e74c3c',
    'Agent 3rd': '#3498db'
}

fig, ax = plt.subplots(figsize=(12, 9))

# 绘制基准线 (OR = 1.0)
ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, zorder=0)

for i, row in df_merged.iterrows():
    # ---------------- 1. 绘制 Human 数据 ----------------
    xerr_h = [[row['OR_Human'] - row['CI_L_Human']], [row['CI_U_Human'] - row['OR_Human']]]
    ax.errorbar(row['OR_Human'], i + offset_h, xerr=xerr_h, 
                fmt='o', color=colors['Human'], capsize=4, elinewidth=2, markersize=8, zorder=3)
    if row['Sig_Human']:
        ax.text(row['CI_U_Human'] + 0.02, i + offset_h, row['Sig_Human'], 
                color=colors['Human'], va='center', fontweight='bold', fontsize=11)

    # ---------------- 2. 绘制 Agent 1st 数据 ----------------
    xerr_a1 = [[row['OR_Agent1st'] - row['CI_L_Agent1st']], [row['CI_U_Agent1st'] - row['OR_Agent1st']]]
    ax.errorbar(row['OR_Agent1st'], i + offset_a1, xerr=xerr_a1, 
                fmt='s', color=colors['Agent 1st'], capsize=4, elinewidth=2, markersize=8, zorder=3)
    if row['Sig_Agent1st']:
        ax.text(row['CI_U_Agent1st'] + 0.02, i + offset_a1, row['Sig_Agent1st'], 
                color=colors['Agent 1st'], va='center', fontweight='bold', fontsize=11)

    # ---------------- 3. 绘制 Agent 3rd 数据 ----------------
    xerr_a3 = [[row['OR_Agent3rd'] - row['CI_L_Agent3rd']], [row['CI_U_Agent3rd'] - row['OR_Agent3rd']]]
    ax.errorbar(row['OR_Agent3rd'], i + offset_a3, xerr=xerr_a3, 
                fmt='^', color=colors['Agent 3rd'], capsize=4, elinewidth=2, markersize=8, zorder=3)
    if row['Sig_Agent3rd']:
        ax.text(row['CI_U_Agent3rd'] + 0.02, i + offset_a3, row['Sig_Agent3rd'], 
                color=colors['Agent 3rd'], va='center', fontweight='bold', fontsize=11)

# ==========================================
# 5. 图表修饰与导出
# ==========================================
ax.set_yticks(y_pos)
ax.set_yticklabels(features, fontsize=12, fontweight='bold')

ax.set_xlabel('Odds Ratio (OR)', fontsize=13, fontweight='bold')
ax.set_title('Persuasion Mechanisms: Human vs. Agent(1st Person) vs. Agent(3rd Person)', fontsize=15, fontweight='bold', pad=20)

# 自定义图例
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Human'], markersize=10, label='Human'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['Agent 1st'], markersize=10, label='Agent (1st Person)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor=colors['Agent 3rd'], markersize=10, label='Agent (3rd Person)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True)

# 坐标轴提示语
ax.text(0.98, -1.0, '← Negative Effect (Less Persuasive)', ha='right', va='center', color='gray', fontsize=11, fontstyle='italic')
ax.text(1.02, -1.0, 'Positive Effect (More Persuasive) →', ha='left', va='center', color='gray', fontsize=11, fontstyle='italic')

# 边框优化
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(axis='x', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
plt.close()

print(f"🎉 任务完成！三维回归对比森林图已保存至: {OUTPUT_IMAGE}")
