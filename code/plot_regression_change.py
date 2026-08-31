"""Figure 7 (corrected) — Change in logistic regression coefficients on textual
features, first-person -> observer.

Forest plot of the nine mechanism features (sorted by human OR ascending):
  - LLM (First-person): red diamond
  - LLM (Observer):     green square
  - Human Reference:    blue vertical tick
  - gray line connects first-person -> observer (the "change")
  - dashed vertical line at OR = 1.0

Point odds ratios (no CI), from the nine-mechanism regressions recomputed from
raw (H1 API-failure branches excluded for the LLM regressions):
  human_nine / agent_nine (first-person) / observer_nine  under regression_output/.
Colors sampled from the reference; no bold text (project rule).
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
RED, GREEN, BLUE = "#df271f", "#00a373", "#2581bc"   # sampled from reference
LINE = "#8a8a8a"

REC = os.path.join(ROOT, "regression_output")
LABELS = {
    "OP Length": "OP Length", "OP Def Freq": "OP Definitive", "OP I Freq": "OP 1stPerson",
    "Ch We Freq": "Reply Inclusive", "Ch Has Formatting": "Reply Formatting",
    "Ch Def Freq": "Reply Definitive", "Ch Dissimilarity": "Reply Dissimilarity",
    "Ch Has Link": "Reply Link", "Ch Length": "Reply Length",
}


def load(path, pooled):
    d = pd.read_csv(path)
    d = d[d["term_type"] == "feature"]
    if pooled:
        d = d[d["regression"] == "pooled_model_fixed_effects"]
    return d.set_index("paper_feature_name")


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


h = load(os.path.join(REC, "human_nine/human_logistic_regression_coefficients_all.csv"), False)
a = load(os.path.join(REC, "agent_nine/agent_logistic_regression_coefficients_all.csv"), True)
o = load(os.path.join(REC, "observer_nine/agent_logistic_regression_coefficients_all.csv"), True)

rows = []
for key, lab in LABELS.items():
    rows.append({
        "Feature": lab,
        "Human": h.loc[key, "odds_ratio"], "Human_p": h.loc[key, "p_value"],
        "First": a.loc[key, "odds_ratio"], "First_p": a.loc[key, "p_value"],
        "Observer": o.loc[key, "odds_ratio"], "Observer_p": o.loc[key, "p_value"],
    })
df = pd.DataFrame(rows).sort_values("Human").reset_index(drop=True)
print(df.round(4).to_string(index=False))

# ---- plot ----
fig, ax = plt.subplots(figsize=(12, 6))
ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.3, zorder=0)
n = len(df)
for i, r in df.iterrows():
    y = n - 1 - i                               # OP Length (row 0) at top
    ax.plot([r.First, r.Observer], [y, y], color=LINE, linewidth=2.0, zorder=1)
    ax.scatter(r.First, y, s=90, marker="D", color=RED, edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter(r.Observer, y, s=90, marker="s", color=GREEN, edgecolor="white", linewidth=0.8, zorder=3)
    ax.plot([r.Human, r.Human], [y - 0.21, y + 0.21], color=BLUE, linewidth=4.0,
            solid_capstyle="butt", zorder=4)

ax.set_yticks(range(n))
ax.set_yticklabels(df["Feature"][::-1], fontsize=23)
ax.set_xlabel("Odds ratio (OR)", fontsize=22)
ax.tick_params(axis="x", labelsize=20)
ax.set_ylim(-0.6, n - 0.4)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

legend = [
    Line2D([0], [0], marker="D", color="w", markerfacecolor=RED, markersize=12, label="LLM (First-person)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=GREEN, markersize=12, label="LLM (Observer)"),
    Line2D([0], [0], marker="|", color=BLUE, linestyle="None", markersize=20, markeredgewidth=3, label="Human Reference"),
]
ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.42, -0.22), ncol=3,
          frameon=False, fontsize=22, columnspacing=0.9, handletextpad=0.35)
plt.tight_layout()
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
out = os.path.join(ROOT, "figures", "regression_change.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print("figure ->", out)
