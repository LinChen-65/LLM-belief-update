"""Figure 1 (corrected) — Discrepancy between LLMs and humans in belief updates.

Reproduces the paper's Fig.1 (stacked FN/FP + Cohen's kappa, first-person) but
aligned to the corrected pipeline:
  - H1: API-failure branches (justification == 'API Error' / error markers) are
        excluded, not counted as a fake delta=False.
  - H2: FP%/FN% denominator = each model's valid (non-failure) branch count,
        not a hardcoded 6032. Metric = share of all valid cases (as in the paper).
Data is read straight from the raw first-person model outputs; no reliance on any
previous intermediate file.

Inputs : data/first_person/final_v3_results_*.json  (raw model judgments)
Outputs: figures/anomalies_and_kappa.png
"""
import json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
FP_DIR = os.path.join(DATA, "first_person")

# 对齐参考图 old_figures/anomalies_and_kappa.png:字体/字重设置
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.sans-serif": ["DejaVu Sans"],
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "axes.unicode_minus": False,
    "font.size": 15,
})
# 颜色取自参考图像素采样(FN 深蓝 / FP 红 / κ 纯绿)
COLOR_FN, COLOR_FP, COLOR_KAPPA = "#123b5d", "#d00100", "#008000"

# paper display name -> raw first-person result file (all 8 models)
FILES = {
    "GPT-4o-mini": "final_v3_results_gpt-4o-mini.json",
    "Gemini-2.5-Flash": "final_v3_results_google_gemini-2.5-flash-lite.json",
    "Qwen-32B": "final_v3_results_Qwen_Qwen2.5-32B-Instruct.json",
    "Qwen-72B": "final_v3_results_Qwen_Qwen2.5-72B-Instruct.json",
    "GLM-4.7": "final_v3_results_Pro_zai-org_GLM-4.7.json",
    "MiniMax-M2.5": "final_v3_results_Pro_MiniMaxAI_MiniMax-M2.5.json",
    "DeepSeek-V3": "final_v3_results_deepseek-ai_DeepSeek-V3.json",
    "GPT-5.5": "final_v3_results_gpt-5.5.json",
}


def is_failure(b):
    # H1: identify API-failure branches
    return bool(b.get("raw_error") or b.get("error")) or \
        (b.get("justification") or "").strip() == "API Error"


def model_metrics(path):
    data = json.load(open(path))
    rows = []  # (human_label, pred 0/1) over valid, non-failure branches
    for e in data:
        for bkey, hlab in [("branch_A_human_success", 1), ("branch_B_human_failure", 0)]:
            b = e.get(bkey, {})
            p = b.get("agent_delta")
            if p not in (True, False):          # drop unparseable / None
                continue
            if is_failure(b):                    # H1: drop API failures
                continue
            rows.append((hlab, 1 if p else 0))
    y_t = [h for h, _ in rows]
    y_p = [p for _, p in rows]
    fp = sum(1 for h, p in rows if h == 0 and p == 1)
    fn = sum(1 for h, p in rows if h == 1 and p == 0)
    n = len(rows)                                # H2: denominator = valid count
    return {
        "n_valid": n,
        "FN_count": fn, "FP_count": fp,
        "FN_pct": fn / n * 100, "FP_pct": fp / n * 100, "Total_pct": (fn + fp) / n * 100,
        "Kappa": cohen_kappa_score(y_t, y_p),
    }


# ---- compute per-model metrics from raw data ----
stats = {name: model_metrics(os.path.join(FP_DIR, f)) for name, f in FILES.items()}

# per-model table (sorted by consistency with human = kappa ascending)
inter_df = (pd.DataFrame(stats).T.reset_index().rename(columns={"index": "Model"})
            .sort_values("Kappa").reset_index(drop=True))
print(inter_df[["Model", "n_valid", "FN_pct", "FP_pct", "Total_pct", "Kappa"]].round(3).to_string(index=False))

# ---- plot (style follows src/plot_all_models_fp_fn_kappa.py) ----
order = inter_df["Model"].tolist()             # sorted by kappa ascending (paper order)
fn_pcts = inter_df["FN_pct"].tolist()
fp_pcts = inter_df["FP_pct"].tolist()
kappas = inter_df["Kappa"].tolist()

fig, ax1 = plt.subplots(figsize=(12, 7))
ax1.bar(order, fn_pcts, width=0.72, color=COLOR_FN, label="FN")
ax1.bar(order, fp_pcts, width=0.72, bottom=fn_pcts, color=COLOR_FP, label="FP")
ax1.set_ylabel("Percentage of Total Cases", fontsize=26)
ax1.tick_params(axis="y", labelsize=19)
ax1.set_xticks(range(len(order)))
ax1.set_xticklabels(order, rotation=35, ha="right", fontsize=20)
max_y1 = max(f + p for f, p in zip(fn_pcts, fp_pcts))
ax1.set_ylim(0, max_y1 * 1.26)

ax2 = ax1.twinx()
ax2.plot(order, kappas, color=COLOR_KAPPA, marker="o", markersize=9, linewidth=3,
         markeredgecolor="white", markeredgewidth=1.5, label="Cohen's κ")
ax2.set_ylabel("Cohen's κ", fontsize=26, color=COLOR_KAPPA)
ax2.tick_params(axis="y", colors=COLOR_KAPPA, labelsize=19)
min_k, max_k = min(kappas), max(kappas)
ax2.set_ylim(0, max_k * 1.5)   # 抬高右轴上限,压低 κ 线,与柱顶 total 数值错开避免遮挡

ax1.set_zorder(1); ax2.set_zorder(2); ax2.patch.set_visible(False)
l1, lab1 = ax1.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2, loc="upper center", bbox_to_anchor=(0.5, 1.17),
           ncol=3, fontsize=24, frameon=False)

bbox = dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none")
for i in range(len(order)):
    fn_v, fp_v, k_v = fn_pcts[i], fp_pcts[i], kappas[i]
    tot = fn_v + fp_v
    if fn_v > 0:
        ax1.text(i, fn_v / 2, f"{fn_v:.1f}%", ha="center", va="center",
                 color="white", fontsize=19)
    if fp_v > 0:
        ax1.text(i, fn_v + fp_v * 0.75, f"{fp_v:.1f}%", ha="center", va="center",
                 color="white", fontsize=19)
    ax1.text(i, tot + max_y1 * 0.04, f"{tot:.1f}%", ha="center", va="bottom",
             color="black", fontsize=19)
    # κ 数值放到点的右上方,与柱顶居中的 total 标注水平错开,避免遮挡
    ax2.annotate(f"{k_v:.3f}", (i, k_v), textcoords="offset points", xytext=(10, 5),
                 ha="left", va="bottom", color=COLOR_KAPPA, fontsize=16, bbox=bbox)

plt.tight_layout()
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
out = os.path.join(ROOT, "figures", "anomalies_and_kappa.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print("figure ->", out)
