"""Figure 9 (corrected) — persuasion strategy effectiveness: first-person vs observer.

Horizontal dumbbell per strategy combination:
  - Observer persuasion rate (blue circle, lower)
  - First-person persuasion rate (red circle, higher)
  - gray line connecting them, with Delta = First-person - Observer annotated
Sorted by first-person rate descending.

Persuasion rate per combo = mean across the 8 models of each model's rate
(fraction of that combo's replies the model judged as delta-awarded).
Recomputed from raw:
  strategy combos: data/strategy/*.json  (H4 failed/None excluded)
  first-person:    data/first_person/*.json (agent_delta, H1 excluded)
  observer:        data/third_person/*.json (delta_awarded, H1 excluded)
Colors sampled from the reference; no bold text (project rule).
"""
import os, glob, json
import numpy as np
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
RED, BLUE = "#d91513", "#0053b8"     # First-person / Observer (sampled)


def combo_label(ps):
    a = []
    if ps.get("logos") is True:
        a.append("L")
    if ps.get("ethos") is True:
        a.append("E")
    if ps.get("pathos") is True:
        a.append("P")
    if not a:
        return "None"
    return {"L": "Logos", "E": "Ethos", "P": "Pathos"}.get(a[0] if len(a) == 1 else None, "+".join(a))


def ann_fail(ps):
    return bool(ps.get("error")) or all(ps.get(k) is None for k in ("logos", "pathos", "ethos"))


def pred_fail(b):
    return bool(b.get("raw_error") or b.get("error")) or (b.get("justification") or "").strip() == "API Error"


# strategy combo per (pair, branch)
combo_lookup = {}
for f in sorted(glob.glob(os.path.join(DATA, "strategy/single_turn_pairs_with_strategies_part*.json"))):
    for pid, pair in json.load(open(f)).items():
        for br in ("success", "failure"):
            if br not in pair:
                continue
            ps = pair[br].get("persuasion_strategies") or {}
            if not ps or ann_fail(ps):
                continue
            c = combo_label(ps)
            if c != "None":
                combo_lookup[(str(pid), br)] = c

BRANCH = {"branch_A_human_success": "success", "branch_B_human_failure": "failure"}


def combo_rate(pattern, key):
    """mean across models of each model's per-combo persuasion rate."""
    per = {}   # combo -> list of per-model rates
    for path in sorted(glob.glob(os.path.join(DATA, pattern))):
        acc = {}
        for e in json.load(open(path)):
            pid = str(e.get("pair_id"))
            for bk, br in BRANCH.items():
                c = combo_lookup.get((pid, br))
                if c is None:
                    continue
                b = e.get(bk, {})
                p = b.get(key)
                if p not in (True, False) or pred_fail(b):
                    continue
                acc.setdefault(c, []).append(1 if p else 0)
        for c, vals in acc.items():
            per.setdefault(c, []).append(np.mean(vals))
    return {c: float(np.mean(v)) for c, v in per.items()}


first = combo_rate("first_person/*.json", "agent_delta")
obs = combo_rate("third_person/*.json", "delta_awarded")

combos = sorted(first, key=lambda c: first[c], reverse=True)   # first-person desc
df = pd.DataFrame([{"combo": c, "First": first[c], "Observer": obs[c],
                    "Delta": first[c] - obs[c]} for c in combos])
print(df.round(4).to_string(index=False))

# ---- plot ----
n = len(df)
fig, ax = plt.subplots(figsize=(12, 7))
for i, r in df.iterrows():
    y = n - 1 - i                                   # first row (highest) at top
    ax.plot([r.Observer, r.First], [y, y], color="#8a8a8a", linewidth=2.0, zorder=1)
    ax.scatter(r.Observer, y, s=270, color=BLUE, zorder=3)
    ax.scatter(r.First, y, s=270, color=RED, zorder=3)
    ax.text(r.Observer - 0.018, y, f"{r.Observer*100:.1f}%", ha="right", va="center", fontsize=24, color=BLUE)
    ax.text(r.First + 0.018, y, f"{r.First*100:.1f}%", ha="left", va="center", fontsize=24, color=RED)
    ax.text((r.Observer + r.First) / 2, y - 0.34, f"Δ={r.Delta*100:.1f}%",
            ha="center", va="center", fontsize=18, color="#333333")

ax.set_yticks(range(n))
ax.set_yticklabels(df["combo"][::-1], fontsize=26)
ax.set_xlabel("Persuasion Rate", fontsize=28)
ax.set_xlim(0, 0.75)
ax.tick_params(axis="x", labelsize=22)
ax.set_ylim(-0.7, n - 0.3)
ax.grid(axis="x", linestyle="--", color="#d8d8d8", alpha=0.8)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=20, label="First-person"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=20, label="Observer"),
]
ax.legend(handles=legend, loc="lower right", fontsize=23, frameon=True)
plt.tight_layout()
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
out = os.path.join(ROOT, "figures", "strategy_effectiveness_first_person_observer.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print("figure ->", out)
