"""Figure 2 (corrected) — Logistic regression coefficients on textual features.

Human vs LLM (first-person, pooled over 8 models with model fixed effects) odds
ratios for the nine mechanism features, with 95% CI and significance stars,
sorted by human OR ascending.

Aligned to the corrected pipeline: the LLM regression uses the raw first-person
model outputs with H1 API-failure branches excluded (human is unaffected).
Coefficients come from the nine-mechanism regressions recomputed from raw:
  regression_output/human_nine/human_logistic_regression_coefficients_all.csv
  regression_output/agent_nine/agent_logistic_regression_coefficients_all.csv
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from statsmodels.stats.multitest import multipletests

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
COLOR_H, COLOR_L = "#1769d1", "#e31a1c"   # sampled from reference (Human blue / LLM red)

HUMAN_CSV = os.path.join(ROOT, "regression_output", "human_nine", "human_logistic_regression_coefficients_all.csv")
AGENT_CSV = os.path.join(ROOT, "regression_output", "agent_nine", "agent_logistic_regression_coefficients_all.csv")

# internal paper_feature_name -> display label used in the paper figure
LABELS = {
    "OP Length": "OP Length",
    "OP Def Freq": "OP Definitive",
    "OP I Freq": "OP 1stPerson",
    "Ch We Freq": "Reply Inclusive",
    "Ch Has Formatting": "Reply Formatting",
    "Ch Def Freq": "Reply Definitive",
    "Ch Dissimilarity": "Reply Dissimilarity",
    "Ch Has Link": "Reply Link",
    "Ch Length": "Reply Length",
}


def load(path, pooled):
    d = pd.read_csv(path)
    d = d[d["term_type"] == "feature"]
    if pooled:
        d = d[d["regression"] == "pooled_model_fixed_effects"]
    return d.set_index("paper_feature_name")


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


h = load(HUMAN_CSV, False)
a = load(AGENT_CSV, True)

rows = []
for key, label in LABELS.items():
    rows.append({
        "Feature": label,
        "OR_H": h.loc[key, "odds_ratio"], "CIl_H": h.loc[key, "or_ci95_low"],
        "CIu_H": h.loc[key, "or_ci95_high"], "p_H": h.loc[key, "p_value"],
        "OR_L": a.loc[key, "odds_ratio"], "CIl_L": a.loc[key, "or_ci95_low"],
        "CIu_L": a.loc[key, "or_ci95_high"], "p_L": a.loc[key, "p_value"],
    })
df = pd.DataFrame(rows).sort_values("OR_H").reset_index(drop=True)
# Benjamini-Hochberg FDR: each regression = one family of 9 tests (rebuttal S1)
df["q_H"] = multipletests(df["p_H"].values, method="fdr_bh")[1]
df["q_L"] = multipletests(df["p_L"].values, method="fdr_bh")[1]
print(df.round(4).to_string(index=False))

# ---- forest plot (Human blue circle above, LLM red diamond below) ----
fig, ax = plt.subplots(figsize=(10, 8))
ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1.5, zorder=1)
off = 0.18
n = len(df)
for i, r in df.iterrows():
    y = n - 1 - i                       # so smallest human OR (row 0) is at top
    # Human
    ax.errorbar(r.OR_H, y + off, xerr=[[r.OR_H - r.CIl_H], [r.CIu_H - r.OR_H]],
                fmt="o", color=COLOR_H, markersize=9, elinewidth=2.5,
                capsize=5, capthick=2.5, zorder=3)
    s = stars(r.q_H)                      # BH-corrected q for significance
    if s:
        ax.text(r.CIu_H + 0.012, y + off, s, color=COLOR_H, va="center", ha="left", fontsize=20)
    # LLM
    ax.errorbar(r.OR_L, y - off, xerr=[[r.OR_L - r.CIl_L], [r.CIu_L - r.OR_L]],
                fmt="D", color=COLOR_L, markersize=9, elinewidth=2.5,
                capsize=5, capthick=2.5, zorder=3)
    s = stars(r.q_L)                      # BH-corrected q for significance
    if s:
        ax.text(r.CIu_L + 0.012, y - off, s, color=COLOR_L, va="center", ha="left", fontsize=20)

ax.set_yticks(range(n))
ax.set_yticklabels([df.iloc[n - 1 - j]["Feature"] for j in range(n)], fontsize=22)
ax.set_xlabel("Odds Ratio (OR)", fontsize=25)
ax.tick_params(axis="x", labelsize=21)
ax.set_ylim(-0.6, n - 0.4)

legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_H, markersize=13, label="Human"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=COLOR_L, markersize=13, label="LLM"),
]
ax.legend(handles=legend_elems, loc="upper right", fontsize=22, frameon=True)

# x range with margin for stars
xmax = max(df["CIu_H"].max(), df["CIu_L"].max())
xmin = min(df["CIl_H"].min(), df["CIl_L"].min())
ax.set_xlim(xmin - 0.03, xmax + 0.12)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(True)

plt.tight_layout()
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
out = os.path.join(ROOT, "figures", "regression_textual_features.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print("figure ->", out)
