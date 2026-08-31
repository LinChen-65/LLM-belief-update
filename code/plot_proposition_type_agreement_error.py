"""Figure 3 (corrected) — LLM judgment bias across proposition types.

Two panels over proposition types (Fact / Value / Policy), first-person condition:
  (a) LLM-Human Agreement : Cohen's kappa
  (b) Error Composition    : FN / (FN + FP)
Each panel shows a light bar (mean across models), per-model dots, and a diamond
at the mean.

Recomputed straight from raw:
  - first-person model outputs: data/first_person/*.json (8 models)
  - proposition-type labels: data/topic/topic_classification_results.json
  - H1: API-failure branches excluded; predictions read from agent_delta.
"""
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import cohen_kappa_score
from scipy import stats

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
COLORS = {"Fact": "#37ab42", "Value": "#fe7139", "Policy": "#765db8"}


def norm_topic(t):
    if not t:
        return None
    return {"fact": "Fact", "value": "Value", "policy": "Policy"}.get(str(t).strip().lower())


def is_failure(b):
    return bool(b.get("raw_error") or b.get("error")) or \
        (b.get("justification") or "").strip() == "API Error"


def pred_of(b):
    for k in ("agent_delta", "delta_awarded"):
        if k in b:
            v = b.get(k)
            if v in (True, False):
                return 1 if v else 0
    return None


# ---- topic labels: OP id -> proposition type ----
topics = json.load(open(os.path.join(DATA, "topic/topic_classification_results.json")))
topic_map = {}
for it in topics:
    rid = it.get("id") or it.get("root_id")
    tp = norm_topic(it.get("proposition_type"))
    if rid and tp:
        topic_map[rid] = tp

# ---- per model x topic: collect (human, pred), compute kappa / FN / FP ----
records = []
for path in sorted(glob.glob(os.path.join(DATA, "first_person/*.json"))):
    model = os.path.basename(path).replace("final_v3_results_", "").replace(".json", "")
    data = json.load(open(path))
    buckets = {t: [] for t in TOPICS}
    for e in data:
        tp = topic_map.get(e.get("root_id"))
        if tp not in TOPICS:
            continue
        for bkey, hlab in [("branch_A_human_success", 1), ("branch_B_human_failure", 0)]:
            b = e.get(bkey, {})
            if not isinstance(b, dict):
                continue
            p = pred_of(b)
            if p is None or is_failure(b):        # H1 filter
                continue
            h = b.get("human_label", hlab)
            h = 1 if h in (1, True) else 0
            buckets[tp].append((h, p))
    for tp in TOPICS:
        pairs = buckets[tp]
        if not pairs:
            continue
        y_t = [h for h, _ in pairs]
        y_p = [p for _, p in pairs]
        fn = sum(1 for h, p in pairs if h == 1 and p == 0)
        fp = sum(1 for h, p in pairs if h == 0 and p == 1)
        kappa = cohen_kappa_score(y_t, y_p) if len(set(y_p)) > 1 and len(set(y_t)) > 1 else np.nan
        err_comp = fn / (fn + fp) if (fn + fp) > 0 else np.nan
        records.append({"Model": model, "Topic": tp, "Kappa": kappa,
                        "FN": fn, "FP": fp, "Error_Composition": err_comp, "N": len(pairs)})

df = pd.DataFrame(records)
print(df.round(4).to_string(index=False))
print("models per topic:", df.groupby("Topic").size().to_dict())

# ---- plot: two panels ----
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
panels = [("Kappa", "(a) LLM-Human Agreement", "Cohen's κ"),
          ("Error_Composition", "(b) Error Composition", "FN / (FN + FP)")]
for ax, (col, title, ylab) in zip(axes, panels):
    for xi, tp in enumerate(TOPICS):
        vals = df[df["Topic"] == tp][col].dropna().values
        mean = float(np.mean(vals))
        c = COLORS[tp]
        ax.bar(xi, mean, width=0.62, color=c, alpha=0.30, edgecolor=c, linewidth=1.5, zorder=1)
        jit = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(xi + jit, vals, color=c, s=45, alpha=0.9, zorder=2, edgecolor="none")
        ax.scatter(xi, mean, color=c, marker="D", s=170, zorder=3,
                   edgecolor="white", linewidth=1.2)
    ax.set_xticks(range(len(TOPICS)))
    ax.set_xticklabels(TOPICS, fontsize=24)
    ax.set_ylabel(ylab, fontsize=24)
    ax.set_title(title, fontsize=25)
    ax.tick_params(axis="y", labelsize=21)
    ax.grid(axis="y", linestyle="-", alpha=0.25)
    ax.set_axisbelow(True)

axes[0].set_ylim(0, None)
plt.tight_layout()
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
out = os.path.join(ROOT, "figures", "proposition_type_agreement_error.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print("figure ->", out)
