"""Figure 15 (new) — OLS of a continuous belief-change score on the nine textual features.

Rebuttal (Reviewer dsTH, W8): GPT-5.1 scores the degree of belief change in each
model's justification text on a 0-100 scale (validated against human pairwise
ranking, Kendall's tau = 0.889). Using this continuous score as the dependent
variable, an OLS is fit on the same nine textual features.

Here we reproduce that OLS from raw:
  - nine raw features per reply: regression_output/agent_nine/base_reply_feature_dataset_without_agent_labels.csv
  - belief_change_score per (pair, branch, model): data/justification/*_belief_change_scores.json
Design: y = belief_change_score; X = z-standardized nine features + model fixed
effects; cluster-robust SE by pair_id (matching the paper's logistic regressions).
Forest plot: coefficient + 95% CI + significance stars. No bold text (project rule).
"""
import os, glob, json, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

# The clustered covariance yields a NaN SE for one model-dummy (nuisance) column;
# it does NOT affect the nine feature SEs/p-values, so we silence the cosmetic warning.
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
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
BLUE = "#1f77b4"

# nine features, bottom-to-top plotting order aligned with Figure 2
# (Fig2 top->bottom: OP Length, OP Definitive, OP 1stPerson, Reply Inclusive, Reply
#  Formatting, Reply Definitive, Reply Dissimilarity, Reply Link, Reply Length)
FEATURES = ["ch_length", "ch_has_link", "ch_dissimilarity", "ch_def_freq", "ch_has_formatting",
            "ch_we_freq", "op_i_freq", "op_def_freq", "op_length"]
LABELS = {
    "op_length": "OP Length", "op_i_freq": "OP 1stPerson", "op_def_freq": "OP Definitive",
    "ch_length": "Reply Length", "ch_def_freq": "Reply Definitive", "ch_we_freq": "Reply Inclusive",
    "ch_dissimilarity": "Reply Dissimilarity", "ch_has_formatting": "Reply Formatting",
    "ch_has_link": "Reply Link",
}
BRANCH = {"branch_A_human_success": "success", "branch_B_human_failure": "failure"}

# ---- per-reply nine raw features ----
base = pd.read_csv(os.path.join(ROOT, "regression_output", "agent_nine", "base_reply_feature_dataset_without_agent_labels.csv"))
base = base[["pair_id", "condition"] + FEATURES].copy()
base["pair_id"] = base["pair_id"].astype(str)

# ---- belief_change_score per (pair, branch, model) ----
rows = []
for path in sorted(glob.glob(os.path.join(DATA, "justification/*_belief_change_scores.json"))):
    model = os.path.basename(path).replace("_belief_change_scores.json", "")
    for e in json.load(open(path)):
        pid = str(e.get("pair_id"))
        for bk, cond in BRANCH.items():
            b = e.get(bk, {})
            sc = b.get("belief_change_score")
            if not isinstance(sc, (int, float)):
                continue
            if b.get("belief_change_error"):
                continue
            if (b.get("justification") or "").strip() == "API Error":
                continue
            rows.append({"pair_id": pid, "condition": cond, "model": model, "score": float(sc)})
scores = pd.DataFrame(rows)

df = scores.merge(base, on=["pair_id", "condition"], how="inner").dropna(subset=FEATURES)
print(f"rows: {len(df)}  (models {df['model'].nunique()}, replies {df[['pair_id','condition']].drop_duplicates().shape[0]})")

# ---- OLS: z-features + model FE, cluster SE by pair ----
Z = (df[FEATURES] - df[FEATURES].mean()) / df[FEATURES].std(ddof=0)
Z.columns = ["z_" + c for c in FEATURES]
dummies = pd.get_dummies(df["model"], prefix="model", drop_first=True).astype(float)
X = sm.add_constant(pd.concat([Z.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1))
y = df["score"].reset_index(drop=True)
res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["pair_id"].values})

ci = res.conf_int()
out = []
for f in FEATURES:
    t = "z_" + f
    out.append({"Feature": LABELS[f], "coef": res.params[t],
                "ci_low": ci.loc[t, 0], "ci_high": ci.loc[t, 1], "p": res.pvalues[t]})
cdf = pd.DataFrame(out)
print(cdf.round(3).to_string(index=False))


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ---- forest plot (single blue series) ----
fig, ax = plt.subplots(figsize=(12, 6))
ax.axvline(0, color="gray", linestyle="--", linewidth=1.3, zorder=1)
for i, f in enumerate(FEATURES):          # i=0 (OP Length) at bottom
    r = cdf.iloc[i]
    ax.errorbar(r.coef, i, xerr=[[r.coef - r.ci_low], [r.ci_high - r.coef]],
                fmt="o", color=BLUE, markersize=13, elinewidth=3.4, capsize=8, capthick=3.4, zorder=3)
    s = stars(r.p)
    if s:
        ax.text(r.ci_high + 0.15, i, s, color="#222222", va="center", ha="left", fontsize=24)
ax.set_yticks(range(len(FEATURES)))
ax.set_yticklabels([LABELS[f] for f in FEATURES], fontsize=23)
ax.set_xlabel("OLS Coefficient", fontsize=25)
ax.tick_params(axis="x", labelsize=21)
ax.set_ylim(-0.6, len(FEATURES) - 0.4)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
out_png = os.path.join(ROOT, "figures", "continuous_belief_score_textual_features.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close()
print("figure ->", out_png)
