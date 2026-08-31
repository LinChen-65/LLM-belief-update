"""Figure 8 (corrected) — first-person vs observer by proposition type.

Two panels over proposition types (Fact / Value / Policy):
  (a) LLM-Human Agreement : Cohen's kappa
  (b) Error Composition    : FN / (FN + FP)
Each type has two bars: First-person (solid) and Observer (hatched), each the
mean across models, with per-model dots overlaid. Bars/dots are colored by
proposition type (aligned with Fig3); the legend distinguishes only the two
perspectives in black/white.

Recomputed from raw:
  first-person: first_person/*.json  (agent_delta)
  observer:     third_person/*.json  (delta_awarded)
  topics:       topic/topic_classification_results.json
H1 API-failure branches excluded. Colors sampled from the reference; no bold.
"""
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import cohen_kappa_score

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
TOPICS = ["Fact", "Value", "Policy"]
COLORS = {"Fact": "#37ab42", "Value": "#fe7139", "Policy": "#765db8"}  # 对齐 Fig3


def norm_topic(t):
    return {"fact": "Fact", "value": "Value", "policy": "Policy"}.get(str(t).strip().lower()) if t else None


def is_fail(b):
    return bool(b.get("raw_error") or b.get("error")) or (b.get("justification") or "").strip() == "API Error"


def pred_of(b, keys):
    for k in keys:
        if k in b and b[k] in (True, False):
            return 1 if b[k] else 0
    return None


topics = json.load(open(os.path.join(DATA, "topic/topic_classification_results.json")))
topic_map = {}
for it in topics:
    rid = it.get("id") or it.get("root_id")
    tp = norm_topic(it.get("proposition_type"))
    if rid and tp:
        topic_map[rid] = tp


def per_model_topic(pattern, keys, perspective):
    recs = []
    for path in sorted(glob.glob(os.path.join(DATA, pattern))):
        model = os.path.basename(path)
        buckets = {t: [] for t in TOPICS}
        for e in json.load(open(path)):
            tp = topic_map.get(e.get("root_id"))
            if tp not in TOPICS:
                continue
            for bk, hl in [("branch_A_human_success", 1), ("branch_B_human_failure", 0)]:
                b = e.get(bk, {})
                if not isinstance(b, dict):
                    continue
                p = pred_of(b, keys)
                if p is None or is_fail(b):
                    continue
                h = 1 if b.get("human_label", hl) in (1, True) else 0
                buckets[tp].append((h, p))
        for tp in TOPICS:
            pr = buckets[tp]
            if not pr:
                continue
            yt = [h for h, _ in pr]
            yp = [p for _, p in pr]
            fn = sum(1 for h, p in pr if h == 1 and p == 0)
            fp = sum(1 for h, p in pr if h == 0 and p == 1)
            k = cohen_kappa_score(yt, yp) if len(set(yt)) > 1 and len(set(yp)) > 1 else np.nan
            ec = fn / (fn + fp) if (fn + fp) else np.nan
            recs.append({"perspective": perspective, "model": model, "Topic": tp,
                         "Kappa": k, "Error_Composition": ec})
    return recs


df = pd.DataFrame(per_model_topic("first_person/*.json", ["agent_delta"], "First-person")
                  + per_model_topic("third_person/*.json", ["delta_awarded", "agent_delta"], "Observer"))

# ---- plot ----
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
W = 0.38
panels = [("Kappa", "(a) LLM-Human Agreement", "Cohen's κ"),
          ("Error_Composition", "(b) Error Composition", "FN / (FN + FP)")]
for ax, (col, title, ylab) in zip(axes, panels):
    for xi, tp in enumerate(TOPICS):
        color = COLORS[tp]
        for j, persp in enumerate(["First-person", "Observer"]):
            vals = df[(df.Topic == tp) & (df.perspective == persp)][col].dropna().values
            xc = xi + (j - 0.5) * W
            m = float(np.mean(vals))
            if persp == "First-person":
                ax.bar(xc, m, width=W, color=color, alpha=0.35, edgecolor=color, linewidth=1.3, zorder=1)
            else:
                ax.bar(xc, m, width=W, facecolor="white", edgecolor=color, hatch="///", linewidth=1.3, zorder=1)
            jit = rng.uniform(-0.09, 0.09, size=len(vals))
            ax.scatter(xc + jit, vals, color=color, s=26, alpha=0.85, edgecolor="none", zorder=3)
    ax.set_xticks(range(len(TOPICS)))
    ax.set_xticklabels(TOPICS, fontsize=26)
    ax.set_ylabel(ylab, fontsize=26)
    ax.set_title(title, fontsize=29)
    ax.tick_params(axis="y", labelsize=21)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
axes[0].set_ylim(0, None)

legend = [
    Patch(facecolor="0.6", edgecolor="black", label="First-person"),
    Patch(facecolor="white", edgecolor="black", hatch="///", label="Observer"),
]
fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=25, bbox_to_anchor=(0.5, -0.11))
plt.tight_layout(rect=[0, 0.05, 1, 1])
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
out = os.path.join(ROOT, "figures", "perspective_comparison_by_proposition_type.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()

print("figure ->", out)
print(df.groupby(["perspective", "Topic"])[["Kappa", "Error_Composition"]].mean().round(4).to_string())
