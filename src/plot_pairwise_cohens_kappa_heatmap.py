import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------
models = ['GPT-4o-mini', 'Gemini-2.5-Flash', 'Qwen-32B',
          'Qwen-72B', 'GLM-4.7', 'MiniMax-M2.5', 'DeepSeek-V3']

fn = np.array([25.5, 8.0, 19.0, 26.9, 30.9, 21.7, 14.6])
fp = np.array([20.8, 37.5, 25.7, 17.8, 13.4, 21.8, 28.0])
total = np.array([46.3, 45.6, 44.8, 44.7, 44.3, 43.4, 42.6])
kappa = np.array([0.074, 0.089, 0.104, 0.105, 0.114, 0.132, 0.147])

x = np.arange(len(models))
bar_width = 0.55

# -----------------------------
# Figure
# -----------------------------
fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
# fig.patch.set_facecolor("white")
# ax.set_facecolor("white")

# -----------------------------
# Stacked bars
# -----------------------------
bars_fn = ax.bar(x, fn, width=bar_width, color='#123f6d', label='FN')
bars_fp = ax.bar(x, fp, width=bar_width, bottom=fn, color='#ff1010', label='FP')

# -----------------------------
# Left axis formatting
# -----------------------------
ax.set_ylabel('Percentage of Total Cases (%)', fontsize=26)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha='right', fontsize=22)
ax.tick_params(axis='y', labelsize=22, width=1.5, length=8)
ax.tick_params(axis='x', width=1.5, length=8)
ax.set_ylim(0, 56)

# dashed horizontal line
ax.axhline(5, linestyle='--', color='0.7', linewidth=2.5)

# -----------------------------
# Text inside bars
# -----------------------------
for i in range(len(models)):
    ax.text(x[i], fn[i] / 2, f'{fn[i]:.1f}%',
            ha='center', va='center', color='white', fontsize=22)
    ax.text(x[i], fn[i] + fp[i] / 2, f'{fp[i]:.1f}%',
            ha='center', va='center', color='white', fontsize=22)

# Total labels at top
for i in range(len(models)):
    ax.text(x[i], total[i] + 1.0, f'Total: {total[i]:.1f}%',
            ha='center', va='bottom', fontsize=18)

# -----------------------------
# Right axis for kappa
# -----------------------------
ax2 = ax.twinx()
line = ax2.plot(
    x, kappa,
    color='green', linewidth=3.5,
    marker='o', markersize=18,
    markerfacecolor='#16a516',
    markeredgecolor='white', markeredgewidth=2.5,
    label='Kappa'
)

ax2.set_ylabel("Cohen's Kappa Score", fontsize=26, color='green')
ax2.tick_params(axis='y', labelcolor='green', labelsize=22, width=1.5, length=8)
ax2.set_ylim(-0.005, 0.185)

# -----------------------------
# Kappa labels BELOW the markers
# -----------------------------
for i, y in enumerate(kappa):
    ax2.annotate(
        f'{y:.3f}',
        xy=(x[i], y),
        xytext=(0, -18),              # 负值 = 标到点的下方
        textcoords='offset points',
        ha='center',
        va='top',
        fontsize=20,
        color='green',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='green', lw=1.5)
    )

# -----------------------------
# Legend
# -----------------------------
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_handles = [
    Patch(facecolor='#123f6d', edgecolor='#123f6d', label='FN'),
    Patch(facecolor='#ff1010', edgecolor='#ff1010', label='FP'),
    Line2D([0], [0], color='green', marker='o', markersize=16,
           markerfacecolor='#16a516', markeredgecolor='white',
           linewidth=3.5, label='Kappa')
]

ax.legend(handles=legend_handles, loc='upper center',
          bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=28)

# -----------------------------
# Layout and save
# -----------------------------
plt.tight_layout()
plt.savefig('stacked_bar_kappa_labels_below.png', dpi=300, bbox_inches='tight')
plt.show()
