import json
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# ==========================================
# 1. 配置文件路径
# ==========================================
STRATEGY_FILE = '/data7/chenyitong/Winning_Arguments/single_turn_pairs_with_strategies.json'

OUTPUT_DIR = "/data7/chenyitong/Winning_Arguments/final_new_Experiment_Analysis_Results/Strategy_Distribution"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "Strategy_Combinations_Distribution.csv")
OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "Strategy_Combinations_PieChart.png")

# 设置全局学术绘图字体
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 数据解析与组合统计
# ==========================================
def analyze_strategy_combinations():
    if not os.path.exists(STRATEGY_FILE):
        print(f"❌ 找不到策略文件: {STRATEGY_FILE}")
        return None
        
    print(f"📦 正在加载并分析策略分布数据: {STRATEGY_FILE}")
    with open(STRATEGY_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    combination_counter = Counter()
    total_cases = 0
    
    # 遍历所有的 pair
    for pair_id, pair_content in data.items():
        for branch_name in ['success', 'failure']:
            if branch_name in pair_content:
                branch_data = pair_content[branch_name]
                strategies = branch_data.get('persuasion_strategies', {})
                
                if not strategies:
                    continue
                    
                total_cases += 1
                active_strategies = []
                
                # 检查哪些策略是 True
                if strategies.get('logos') is True:
                    active_strategies.append('Logos')
                if strategies.get('pathos') is True:
                    active_strategies.append('Pathos')
                if strategies.get('ethos') is True:
                    active_strategies.append('Ethos')
                    
                # 决定当前案例所属的组合类别
                if not active_strategies:
                    combo_name = 'None'
                else:
                    active_strategies.sort()
                    combo_name = ' + '.join(active_strategies)
                    
                combination_counter[combo_name] += 1
                
    # ==========================================
    # 3. 结果汇总与终端打印
    # ==========================================
    print("\n" + "="*60)
    print(" 📊 说服策略组合分布统计 (Strategy Combinations)")
    print("="*60)
    print(f"总计分析案例数: {total_cases} 条 (回复)")
    print("-" * 60)
    
    results = []
    for combo, count in combination_counter.most_common():
        percentage = (count / total_cases) * 100
        results.append({
            'Strategy_Combination': combo,
            'Count': count,
            'Percentage_in_All(%)': round(percentage, 2)
        })
        print(f" - [{combo:<30}]: {count:>4} 条 (占整体: {percentage:>5.2f}%)")
        
    print("="*60 + "\n")
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    return df, total_cases

# ==========================================
# 4. 绘制分布扇形图 (剔除 None)
# ==========================================
def plot_distribution(df, total_cases):
    if df is None or df.empty:
        return
        
    print("🎨 正在绘制策略分布扇形图 (已过滤 None)...")
    
    # 【修改点】：直接在绘图前过滤掉 'None' 分类
    df_filtered = df[df['Strategy_Combination'] != 'None']
    
    # 按照出现频率降序，饼图从面积最大开始画
    df_sorted = df_filtered.sort_values(by='Count', ascending=False)
    
    labels = df_sorted['Strategy_Combination'].tolist()
    counts = df_sorted['Count'].tolist()
    
    # 计算有效策略的总数 (不含 None)
    valid_cases = sum(counts)
    
    # 准备扇区颜色（使用 Set3 色板提供较高的对比度和区分度）
    cmap = plt.get_cmap('Set3')
    colors = [cmap(i % 12) for i in range(len(labels))]
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # 定义在扇形图上显示比例及数值的函数
    def my_autopct(pct):
        val = int(round(pct * valid_cases / 100.0))
        # 当扇区过小 (占比小于 2%) 时，隐藏文字以防止重叠
        return f'{pct:.1f}%\n({val})' if pct > 2.0 else ''

    # 绘制扇形图
    wedges, texts, autotexts = ax.pie(
        counts, 
        labels=labels, 
        colors=colors,
        autopct=my_autopct,
        startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
        textprops={'fontsize': 11, 'color': '#333333'}
    )
    
    # 加粗扇区内的百分比和数值文字
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # 标题更新：说明基数是不含 None 的有效策略数
    ax.set_title(f'Distribution of Persuasion Strategy Combinations\n(Excluding "None" | Meaningful Cases: {valid_cases})', 
                 fontsize=15, fontweight='bold', pad=20)
    
    ax.legend(wedges, labels,
              title="Strategy Combinations",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, bbox_inches='tight')
    plt.close()
    
    print(f"🎉 绘图完成！")
    print(f"📈 本次饼图基于 {valid_cases} 个包含至少一种策略的有效案例绘制。")
    print(f"📊 分布扇形图已保存至: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    result_df, total_count = analyze_strategy_combinations()
    if result_df is not None:
        plot_distribution(result_df, total_count)
