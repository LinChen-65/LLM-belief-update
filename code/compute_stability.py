"""Appendix B, Table 1 — Judgment Stability across Repeated Runs.

For each model we have three first-person runs (temperature = 0.1):
  - main experiment : data/first_person/final_v3_results_<fid>.json
  - seed 1, seed 2  : data/random_seed/<...>.json
Prediction field = agent_delta. A branch is valid if agent_delta is boolean and
the branch is not an API failure (raw_error/error truthy or justification=="API Error").

Metrics per model, over branches valid in ALL three runs (intersection):
  - Fleiss' kappa across the three runs (2 categories: delta True/False)
  - Pairwise agreement: min over the three run-pairs (reported as ">= x%")
  - Three-run exact match: fraction of branches where all three runs agree

Validation: the Qwen2.5-72B and Gemini-2.5-Flash rows reproduce the values already
reported in the rebuttal (Fleiss' kappa 0.888 / 0.887).

All eight models now have three complete runs (MiniMax-M2.5's seed2 was re-run;
earlier it had failed on ~63% of branches).
"""
import os, json, itertools
import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

MAIN = os.path.join(DATA, "first_person/final_v3_results_{}.json")
SEED = os.path.join(DATA, "random_seed/{}")

# display -> (main_fid, seed1_file, seed2_file)
MODELS = {
    "GPT-4o-mini": ("gpt-4o-mini", "final_v3_results_gpt-4o-mini_v3_seed1.json", "final_v3_results_gpt-4o-mini_v3_seed2.json"),
    "Qwen2.5-72B-Instruct": ("Qwen_Qwen2.5-72B-Instruct", "qwen_seed_1_results.json", "qwen_seed_2_results.json"),
    "DeepSeek-V3": ("deepseek-ai_DeepSeek-V3", "final_v3_results_deepseek-ai_DeepSeek-V3_seed1.json", "final_v3_results_deepseek-ai_DeepSeek-V3_seed2.json"),
    "Gemini-2.5-Flash": ("google_gemini-2.5-flash-lite", "final_seed1_results_google_gemini-2.5-flash-lite.json", "final_seed2_results_google_gemini-2.5-flash-lite.json"),
    "Qwen2.5-32B-Instruct": ("Qwen_Qwen2.5-32B-Instruct", "final_v3_results_Qwen_Qwen2.5-32B-Instruct_seed1.json", "final_v3_results_Qwen_Qwen2.5-32B-Instruct_seed2.json"),
    "GPT-5.5": ("gpt-5.5", "final_v3_results_gpt-5.5_seed1.json", "final_v3_results_gpt-5.5_seed2.json"),
    "GLM-4.7": ("Pro_zai-org_GLM-4.7", "final_v3_results_z-ai_glm-4.7_seed1.json", "final_v3_results_z-ai_glm-4.7_seed2.json"),
    "MiniMax-M2.5": ("Pro_MiniMaxAI_MiniMax-M2.5", "final_v3_results_Pro_MiniMaxAI_MiniMax-M2.5_seed1.json", "final_v3_results_Pro_MiniMaxAI_MiniMax-M2.5_seed2.json"),
}
BLANK = []   # all eight models now have three complete runs


def is_fail(b):
    return bool(b.get("raw_error") or b.get("error")) or (b.get("justification") or "").strip() == "API Error"


def load(path):
    m = {}
    for e in json.load(open(path)):
        pid = e.get("pair_id")
        for bk in ("branch_A_human_success", "branch_B_human_failure"):
            b = e.get(bk, {})
            v = b.get("agent_delta")
            if v in (True, False) and not is_fail(b):
                m[(pid, bk)] = 1 if v else 0
    return m


rows = []
for name, (fid, s1, s2) in MODELS.items():
    r0, r1, r2 = load(MAIN.format(fid)), load(SEED.format(s1)), load(SEED.format(s2))
    keys = sorted(set(r0) & set(r1) & set(r2))
    A = np.array([[r0[k], r1[k], r2[k]] for k in keys])
    tbl = np.zeros((len(keys), 2), int)
    ones = A.sum(axis=1)
    tbl[:, 1] = ones
    tbl[:, 0] = 3 - ones
    fk = fleiss_kappa(tbl)
    pair = [float(np.mean(A[:, i] == A[:, j])) for i, j in itertools.combinations(range(3), 2)]
    exact = float(np.mean([len(set(row)) == 1 for row in A]))
    rows.append({"Model": name, "Fleiss_kappa": fk, "Min_pairwise_agreement": min(pair),
                 "Three_run_exact_match": exact, "n": len(keys)})

df = pd.DataFrame(rows).sort_values("Fleiss_kappa", ascending=False).reset_index(drop=True)
print(df.round(3).to_string(index=False))
