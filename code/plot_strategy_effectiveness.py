"""Figure 5 (corrected) — Effectiveness of persuasion strategies.

Scatter of human vs first-person LLM persuasion rate per strategy combination:
  - x = Persuasion Rate (Human), y = Persuasion Rate (First-person)
  - big diamond per combo = (human rate, mean LLM rate across models)
  - small dots = per-model LLM rate at that combo's human rate
  - upper-left "LLM > Human" (blue), lower-right "LLM < Human" (red), y=x dashed

Aligned to the corrected pipeline:
  - strategy combos from raw annotations with H4 failed/None annotations excluded
  - first-person predictions from raw with H1 API-failure branches excluded
  - all 8 first-person models
Style follows src/plot_persuasion_strategy_human_vs_first_person.py (colors etc.);
per the project rule, no bold text.
"""
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

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
COLOR_MAP = {
    "Logos": "#2E86DE", "Pathos": "#8E5B4C", "Ethos": "#F1949B",
    "L+P": "#F39C12", "L+E": "#27AE60", "E+P": "#7D6BC2", "L+E+P": "#E74C3C",
}


def combo_label(ps):
    active = []
    if ps.get("logos") is True:
        active.append("L")
    if ps.get("ethos") is True:
        active.append("E")
    if ps.get("pathos") is True:
        active.append("P")
    if not active:
        return "None"
    if active == ["L"]:
        return "Logos"
    if active == ["E"]:
        return "Ethos"
    if active == ["P"]:
        return "Pathos"
    return "+".join(active)


def is_ann_fail(ps):
    return bool(ps.get("error")) or all(ps.get(k) is None for k in ("logos", "pathos", "ethos"))


def is_pred_fail(b):
    return bool(b.get("raw_error") or b.get("error")) or \
        (b.get("justification") or "").strip() == "API Error"


# ---- strategy combo per (pair, branch) ----  (branch: success / failure)
combo_lookup = {}
for f in sorted(glob.glob(os.path.join(DATA, "strategy/single_turn_pairs_with_strategies_part*.json"))):
    d = json.load(open(f))
    for pid, pair in d.items():
        for br in ("success", "failure"):
            if br not in pair:
                continue
            ps = pair[br].get("persuasion_strategies") or {}
            if not ps or is_ann_fail(ps):        # H4: drop failed annotations
                continue
            c = combo_label(ps)
            if c != "None":
                combo_lookup[(str(pid), br)] = c

BRANCH = {"branch_A_human_success": ("success", 1), "branch_B_human_failure": ("failure", 0)}

# ---- reply-level records: combo, human_label, per-model first-person pred ----
rows = []
for path in sorted(glob.glob(os.path.join(DATA, "first_person/*.json"))):
    model = os.path.basename(path).replace("final_v3_results_", "").replace(".json", "")
    for e in json.load(open(path)):
        pid = str(e.get("pair_id"))
        for bkey, (br, hlab) in BRANCH.items():
            c = combo_lookup.get((pid, br))
            if c is None:
                continue
            b = e.get(bkey, {})
            p = b.get("agent_delta")
            if p not in (True, False) or is_pred_fail(b):   # H1: drop failures
                continue
            h = b.get("human_label", hlab)
            rows.append({"model": model, "combo": c,
                         "human": 1 if h in (1, True) else 0, "pred": 1 if p else 0,
                         "key": (pid, br)})

df = pd.DataFrame(rows)

# human persuasion rate per combo (dedup on pair+branch), model LLM rate per combo
human_rate = (df.drop_duplicates(["key", "combo"]).groupby("combo")["human"].mean())
model_rate = df.groupby(["combo", "model"])["pred"].mean().reset_index(name="llm_rate")
model_rate["human_rate"] = model_rate["combo"].map(human_rate)
mean_rate = model_rate.groupby("combo").agg(human_rate=("human_rate", "first"),
                                            llm_rate_mean=("llm_rate", "mean")).reset_index()
print(mean_rate.round(4).to_string(index=False))

# ---- plot ----
combos = list(COLOR_MAP.keys())
fig, ax = plt.subplots(figsize=(8, 8))
allx = model_rate["human_rate"].tolist() + mean_rate["human_rate"].tolist()
ally = model_rate["llm_rate"].tolist() + mean_rate["llm_rate_mean"].tolist()
xlow, xhigh = max(0.0, min(allx) - 0.05), min(1.0, max(allx) + 0.13)
ylow, yhigh = max(0.0, min(ally) - 0.05), min(1.0, max(ally) + 0.05)
lo, hi = min(xlow, ylow), max(xhigh, yhigh)
xs = np.linspace(xlow, xhigh, 400)
ax.fill_between(xs, xs, yhigh, color="#7aa6d8", alpha=0.10, zorder=0)
ax.fill_between(xs, ylow, xs, color="#d99b9b", alpha=0.10, zorder=0)
ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.0, alpha=0.7, zorder=1)

for c in combos:
    sub = model_rate[model_rate["combo"] == c]
    ax.scatter(sub["human_rate"], sub["llm_rate"], s=40, color=COLOR_MAP[c],
               alpha=0.40, edgecolor="none", zorder=2)
# 每个组合标签的定制偏移 (dx, dy, ha, va),避免与点及相邻标签重叠
LABEL_POS = {
    "Logos":  (0.013, 0.015, "left", "bottom"),
    "E+P":    (-0.013, 0.015, "right", "bottom"),
    "L+P":    (0.013, 0.013, "left", "bottom"),
    "L+E":    (0.015, -0.004, "left", "center"),
    "L+E+P":  (0.015, 0.012, "left", "bottom"),
    "Ethos":  (0.013, 0.013, "left", "bottom"),
    "Pathos": (0.013, -0.012, "left", "top"),
}
for _, r in mean_rate.iterrows():
    c = r["combo"]
    ax.scatter(r["human_rate"], r["llm_rate_mean"], s=200, marker="D",
               color=COLOR_MAP[c], edgecolor="white", linewidth=1.5, zorder=4)
    dx, dy, ha, va = LABEL_POS[c]
    ax.text(r["human_rate"] + dx, r["llm_rate_mean"] + dy, c,
            fontsize=22, color=COLOR_MAP[c], ha=ha, va=va, zorder=6,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])

ax.text(xlow + 0.01, yhigh - 0.01, "LLM > Human", color="#5b8fd0", fontsize=23, va="top")
ax.text(xhigh - 0.01, ylow + 0.01, "LLM < Human", color="#d98880", fontsize=23, ha="right", va="bottom")
ax.set_xlim(xlow, xhigh)
ax.set_ylim(ylow, yhigh)
ax.set_xlabel("Persuasion Rate (Human)", fontsize=21)
ax.set_ylabel("Persuasion Rate (First-person)", fontsize=21)
ax.tick_params(labelsize=16)
plt.tight_layout()
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
out = os.path.join(ROOT, "figures", "strategy_effectiveness.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print("figure ->", out)
