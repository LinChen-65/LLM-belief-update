"""Figure 17 (corrected) — within-model consistency between first-person and
observer judgments.

For each model, compare its first-person prediction (agent_delta) and observer
prediction (delta_awarded) on the same branches (where both are valid, H1
failures excluded):
  - Raw agreement : fraction of branches where the two perspectives agree
  - Cohen's kappa : chance-corrected agreement between the two perspectives
Grouped bar chart, models sorted by raw agreement descending.

Recomputed from raw:
  data/first_person/*.json  (agent_delta)
  data/third_person/*.json  (delta_awarded)
Colors sampled from the reference; no bold text (project rule).
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
BLUE, ORANGE = "#2177b6", "#fe780f"      # Raw agreement / Cohen's kappa (sampled)

MODELS = {
    "GLM-4.7": "Pro_zai-org_GLM-4.7", "Qwen-72B": "Qwen_Qwen2.5-72B-Instruct",
    "GPT-4o-mini": "gpt-4o-mini", "MiniMax-M2.5": "Pro_MiniMaxAI_MiniMax-M2.5",
    "DeepSeek-V3": "deepseek-ai_DeepSeek-V3", "Qwen-32B": "Qwen_Qwen2.5-32B-Instruct",
    "GPT-5.5": "gpt-5.5", "Gemini-2.5-Flash": "google_gemini-2.5-flash-lite",
}


def is_fail(b):
    return bool(b.get("raw_error") or b.get("error")) or (b.get("justification") or "").strip() == "API Error"


def load(path, key):
    m = {}
    for e in json.load(open(path)):
        pid = e.get("pair_id")
        for bk in ("branch_A_human_success", "branch_B_human_failure"):
            b = e.get(bk, {})
            p = b.get(key)
            if p in (True, False) and not is_fail(b):
                m[(pid, bk)] = 1 if p else 0
    return m


rows = []
for name, fid in MODELS.items():
    fp = load(os.path.join(DATA, f"first_person/final_v3_results_{fid}.json"), "agent_delta")
    ob = load(os.path.join(DATA, f"third_person/final_v3_results_observer_{fid}.json"), "delta_awarded")
    keys = [k for k in fp if k in ob]
    a = [fp[k] for k in keys]
    b = [ob[k] for k in keys]
    rows.append({"Model": name, "Raw_agreement": float(np.mean([x == y for x, y in zip(a, b)])),
                 "Cohen_kappa": cohen_kappa_score(a, b), "n": len(keys)})

df = pd.DataFrame(rows).sort_values("Raw_agreement", ascending=False).reset_index(drop=True)
print(df.round(3).to_string(index=False))

# ---- grouped bar chart ----
order = df["Model"].tolist()
raw = df["Raw_agreement"].tolist()
kap = df["Cohen_kappa"].tolist()
x = np.arange(len(order))
w = 0.4
fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar(x - w / 2, raw, width=w, color=BLUE, label="Raw agreement")
b2 = ax.bar(x + w / 2, kap, width=w, color=ORANGE, label="Cohen's kappa")
for bars, vals in [(b1, raw), (b2, kap)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.2f}",
                ha="center", va="bottom", fontsize=20)
ax.set_ylabel("Agreement", fontsize=25)
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=25, ha="right", fontsize=22)
ax.tick_params(axis="y", labelsize=20)
ax.set_ylim(0, 1.18)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.legend(fontsize=22, frameon=True, loc="upper right")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
out = os.path.join(ROOT, "figures", "perspective_consistency.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print("figure ->", out)
