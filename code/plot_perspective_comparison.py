"""Figure 6 (corrected) — First-person vs Observer error composition (FN / FP).

Two-panel dumbbell: for each model, the first-person (circle) and observer
(diamond) FN% (left panel) and FP% (right panel), as a share of all valid cases.

Aligned to the corrected pipeline: H1 API-failure branches excluded; denominator
is each model's valid branch count (H2), i.e. "% of total cases" as in Fig.1.
Recomputed from raw:
  first_person/*.json   (agent_delta)
  third_person/*.json   (delta_awarded)
Colors sampled from the reference; no bold text (project rule).
"""
import os, glob, json
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
FN_FIRST, FN_OBS = "#2f80ed", "#0b57d0"     # sampled: blue / dark blue
FP_FIRST, FP_OBS = "#ff4d00", "#e60000"     # sampled: orange / red
LINE = "#888888"

NAME = {
    "gpt-4o-mini": "GPT-4o-mini", "google_gemini-2.5-flash-lite": "Gemini-2.5-Flash",
    "Qwen_Qwen2.5-32B-Instruct": "Qwen2.5-32B", "Qwen_Qwen2.5-72B-Instruct": "Qwen2.5-72B",
    "Pro_zai-org_GLM-4.7": "GLM-4.7", "Pro_MiniMaxAI_MiniMax-M2.5": "MiniMax-M2.5",
    "deepseek-ai_DeepSeek-V3": "DeepSeek-V3", "gpt-5.5": "GPT-5.5",
}
ORDER = ["Qwen2.5-72B", "Qwen2.5-32B", "MiniMax-M2.5", "Gemini-2.5-Flash",
         "GPT-4o-mini", "GLM-4.7", "DeepSeek-V3", "GPT-5.5"]


def is_fail(b):
    return bool(b.get("raw_error") or b.get("error")) or \
        (b.get("justification") or "").strip() == "API Error"


def fn_fp_pct(path, key):
    rows = []
    for e in json.load(open(path)):
        for bk, hl in [("branch_A_human_success", 1), ("branch_B_human_failure", 0)]:
            b = e.get(bk, {})
            p = b.get(key)
            if p not in (True, False) or is_fail(b):
                continue
            rows.append((hl, 1 if p else 0))
    n = len(rows)
    fn = sum(1 for h, p in rows if h == 1 and p == 0)
    fp = sum(1 for h, p in rows if h == 0 and p == 1)
    return fn / n * 100, fp / n * 100


def short(path):
    b = os.path.basename(path).replace("final_v3_results_observer_", "").replace("final_v3_results_", "").replace(".json", "")
    return NAME.get(b, b)


first = {short(f): fn_fp_pct(f, "agent_delta") for f in glob.glob(os.path.join(DATA, "first_person/*.json"))}
obs = {short(f): fn_fp_pct(f, "delta_awarded") for f in glob.glob(os.path.join(DATA, "third_person/*.json"))}

df = pd.DataFrame([{
    "Model": m,
    "FN_first": first[m][0], "FN_obs": obs[m][0],
    "FP_first": first[m][1], "FP_obs": obs[m][1],
} for m in ORDER])
print(df.round(2).to_string(index=False))

# ---- plot ----
fig, axes = plt.subplots(1, 2, figsize=(19, 9), sharey=True)
panels = [("FN_first", "FN_obs", FN_FIRST, FN_OBS, "False Negative (FN)"),
          ("FP_first", "FP_obs", FP_FIRST, FP_OBS, "False Positive (FP)")]
n = len(df)
for ax, (c1, c2, col1, col2, title) in zip(axes, panels):
    for i, r in df.iterrows():
        y = n - 1 - i                       # first row at top
        v1, v2 = r[c1], r[c2]
        ax.plot([v1, v2], [y, y], color=LINE, linewidth=2.0, zorder=1)
        ax.scatter(v1, y, s=300, marker="o", color=col1, edgecolor="white", linewidth=1.2, zorder=3)
        ax.scatter(v2, y, s=150, marker="D", color=col2, edgecolor="white", linewidth=1.2, zorder=3)
        # value labels on the outer side of each end
        lo = (v1, col1) if v1 <= v2 else (v2, col2)
        hi = (v2, col2) if v2 >= v1 else (v1, col1)
        ax.text(lo[0] - 1.8, y, f"{lo[0]:.1f}%", ha="right", va="center", fontsize=26, color=lo[1])
        ax.text(hi[0] + 1.8, y, f"{hi[0]:.1f}%", ha="left", va="center", fontsize=26, color=hi[1])
    ax.set_yticks(range(n))
    ax.set_yticklabels(ORDER[::-1], fontsize=30)
    ax.set_title(title, fontsize=35)
    ax.set_xlabel("Percentage of Total Cases (%)", fontsize=29)
    ax.set_xlim(0, 52)
    ax.tick_params(axis="x", labelsize=26)
    ax.grid(axis="x", linestyle="--", color="#d8d8d8", alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#333333", markersize=15, label="First-person"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#888888", markersize=10.6, label="Observer"),
]
fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False, fontsize=30, bbox_to_anchor=(0.5, 1.10))
plt.tight_layout()
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
out = os.path.join(ROOT, "figures", "perspective_comparison.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print("figure ->", out)
