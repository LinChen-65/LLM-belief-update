"""Figure 4 (corrected) — Distribution of persuasion strategies in the dataset.

Pie chart of persuasion-strategy combinations over strategy-bearing replies.
Aligned to the corrected pipeline (H4): failed strategy annotations (error field
or all-None logos/pathos/ethos) are excluded, not counted as "None".

Recomputed from raw strategy annotations:
  data/strategy/single_turn_pairs_with_strategies_part{1,2}.json
Style follows the paper's Fig.4: labels outside, percentages inside large wedges,
leader lines for small wedges, no legend/title. Colors sampled from the reference.
"""
import os, glob, json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.sans-serif": ["DejaVu Sans"],
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "axes.unicode_minus": False,
})
# colors in frequency-descending order, sampled from the reference figure
PALETTE = ["#6ac6b8", "#fecf4c", "#b7aad5", "#fe4340", "#459edb", "#fe7c0d", "#62b22d"]


def combo(ps):
    active = [n for n, k in [("Logos", "logos"), ("Pathos", "pathos"), ("Ethos", "ethos")]
              if ps.get(k) is True]
    return "None" if not active else " + ".join(sorted(active))


# ---- corrected distribution over strategy-bearing replies (H4) ----
cnt = Counter()
for f in sorted(glob.glob(os.path.join(DATA, "strategy/single_turn_pairs_with_strategies_part*.json"))):
    d = json.load(open(f))
    for pid, pair in d.items():
        for br in ("success", "failure"):
            if br not in pair:
                continue
            ps = pair[br].get("persuasion_strategies") or {}
            if not ps:
                continue
            # H4: skip failed annotations (error field or all-None)
            if ps.get("error") or all(ps.get(k) is None for k in ("logos", "pathos", "ethos")):
                continue
            cnt[combo(ps)] += 1

cnt.pop("None", None)                      # pie is over strategy-bearing replies
items = cnt.most_common()                  # sorted by count desc
labels = [k for k, _ in items]
counts = [v for _, v in items]
total = sum(counts)
pcts = [c / total * 100 for c in counts]

print("meaningful cases:", total)
for lab, c, p in zip(labels, counts, pcts):
    print(f"  {lab:26} {c:5}  {p:5.1f}%")

# ---- pie ----
fig, ax = plt.subplots(figsize=(9, 7))
colors = PALETTE[:len(labels)]
wedges, _ = ax.pie(counts, colors=colors, startangle=112,
                   wedgeprops={"edgecolor": "white", "linewidth": 1.5})

BIG = 3.0   # % threshold: big wedge (name outside + % inside) vs small (leader line)
small = []
for w, lab, p in zip(wedges, labels, pcts):
    ang = np.deg2rad((w.theta1 + w.theta2) / 2)
    x, y = np.cos(ang), np.sin(ang)
    if p >= BIG:
        ha = "left" if x >= 0 else "right"
        ax.text(0.62 * x, 0.62 * y, f"{p:.1f}%", ha="center", va="center",
                fontsize=23, color="#222222")
        ax.text(1.12 * x, 1.12 * y, lab, ha=ha, va="center", fontsize=23)
    else:
        small.append((y, x, lab, p))   # collect small wedges for staggered labels

# small wedges: stack their labels on the upper-left, vertically staggered, with leader lines
small.sort(key=lambda t: t[0], reverse=True)   # top-most wedge first
y_top = 1.12
for j, (wy, wx, lab, p) in enumerate(small):
    ty = y_top - j * 0.22
    ax.annotate(f"{lab}\n{p:.1f}%",
                xy=(wx, wy), xytext=(-1.15, ty),
                ha="right", va="center", fontsize=21,
                arrowprops=dict(arrowstyle="-", color="gray", lw=1))

ax.set_aspect("equal")
plt.tight_layout()
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
out = os.path.join(ROOT, "figures", "distribution_strategy.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print("figure ->", out)
