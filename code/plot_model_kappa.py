"""Figure 14 (corrected) — pairwise inter-model Cohen's kappa (first-person).

Lower-triangular heatmap of Cohen's kappa between every pair of models'
first-person delta judgments (agent_delta). Diagonal = 1.0.

Pairwise kappa is computed over the branches where BOTH models have a valid,
non-failure prediction (H1 API-failure branches excluded per pair).
Recomputed from raw: data/first_person/*.json.
Colormap = red half of RdBu_r over [0,1] (colors identical to RdBu_r for kappa>=0). No bold text.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
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

FILES = {
    "DeepSeek-V3": "final_v3_results_deepseek-ai_DeepSeek-V3.json",
    "Gemini-2.5-Flash": "final_v3_results_google_gemini-2.5-flash-lite.json",
    "GPT-4o-mini": "final_v3_results_gpt-4o-mini.json",
    "MiniMax-M2.5": "final_v3_results_Pro_MiniMaxAI_MiniMax-M2.5.json",
    "GLM-4.7": "final_v3_results_Pro_zai-org_GLM-4.7.json",
    "Qwen-32B": "final_v3_results_Qwen_Qwen2.5-32B-Instruct.json",
    "Qwen-72B": "final_v3_results_Qwen_Qwen2.5-72B-Instruct.json",
    "GPT-5.5": "final_v3_results_gpt-5.5.json",
}
NAMES = list(FILES)


def is_fail(b):
    return bool(b.get("raw_error") or b.get("error")) or (b.get("justification") or "").strip() == "API Error"


# per-branch prediction per model (NaN when missing/failure)
cols = {}
for name, fn in FILES.items():
    m = {}
    for e in json.load(open(os.path.join(DATA, "first_person", fn))):
        pid = e.get("pair_id")
        for bk in ("branch_A_human_success", "branch_B_human_failure"):
            b = e.get(bk, {})
            p = b.get("agent_delta")
            m[(pid, bk)] = (1 if p else 0) if (p in (True, False) and not is_fail(b)) else np.nan
    cols[name] = m
mat = pd.DataFrame(cols)

K = pd.DataFrame(index=NAMES, columns=NAMES, dtype=float)
for a in NAMES:
    for b in NAMES:
        if a == b:
            K.loc[a, b] = 1.0
        else:
            sub = mat[[a, b]].dropna()
            K.loc[a, b] = cohen_kappa_score(sub[a], sub[b])
print(K.round(3).to_string())

# ---- heatmap (lower triangle) ----
M = K.values.astype(float)
masked = np.ma.masked_where(np.triu(np.ones_like(M, dtype=bool), k=1), M)
fig, ax = plt.subplots(figsize=(9, 7.5))
_red = mcolors.LinearSegmentedColormap.from_list("rdbu_red", plt.cm.RdBu_r(np.linspace(0.5, 1.0, 256)))
im = ax.imshow(masked, cmap=_red, vmin=0, vmax=1)
ax.set_xticks(range(len(NAMES)))
ax.set_yticks(range(len(NAMES)))
ax.set_xticklabels(NAMES, rotation=45, ha="right", fontsize=19)
ax.set_yticklabels(NAMES, fontsize=19)
for i in range(len(NAMES)):
    for j in range(len(NAMES)):
        if j <= i:
            val = M[i, j]
            tc = "white" if val >= 0.999 else "black"   # 仅对角线(1.0)白字,其余黑字
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=13, color=tc)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Cohen's κ", fontsize=21)
cb.ax.tick_params(labelsize=17)
plt.tight_layout()
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
out = os.path.join(ROOT, "figures", "model_kappa.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print("figure ->", out)
