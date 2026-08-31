"""Benjamini-Hochberg FDR correction for Table 2 (the full 40 textual features).

Same design as the nine-feature correction, extended to Tan et al. (2016)'s full
set of 40 textual features (paper Table 2 / Appendix E). Each regression is one
family of 40 tests, corrected with multipletests(method="fdr_bh").

Families (corrected regressions recomputed from raw, H1 excluded):
  - Human            : regression_output/human/human_logistic_regression_coefficients_all.csv
  - LLM-first-person : regression_output/agent/agent_logistic_regression_coefficients_all.csv  (pooled + model FE)
  - LLM-observer     : regression_output/observer/agent_logistic_regression_coefficients_all.csv (pooled + model FE)
Trend = sign of coef_logit (up / down); p = raw p-value; q = BH-adjusted.
Prints the per-family significance summary to stdout.
"""
import os
import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REC = os.path.join(ROOT, "regression_output")
FAMILIES = {
    "Human": ("human/human_logistic_regression_coefficients_all.csv", False),
    "LLM-first-person": ("agent/agent_logistic_regression_coefficients_all.csv", True),
    "LLM-observer": ("observer/agent_logistic_regression_coefficients_all.csv", True),
}


def load(path, pooled):
    d = pd.read_csv(path)
    d = d[d["term_type"] == "feature"]
    if pooled:
        d = d[d["regression"] == "pooled_model_fixed_effects"]
    return d.set_index("feature_id")[["coef_logit", "p_value", "term"]]


data = {fam: load(os.path.join(REC, rel), pooled) for fam, (rel, pooled) in FAMILIES.items()}
feats = list(data["Human"].index)   # 40 features, human order

# BH per family (40 tests each)
q = {}
for fam, d in data.items():
    pv = [float(d.loc[f, "p_value"]) for f in feats]
    q[fam] = dict(zip(feats, multipletests(pv, method="fdr_bh")[1]))


print(f"features: {len(feats)}")
for fam, d in data.items():
    nraw = sum(1 for f in feats if float(d.loc[f, 'p_value']) < 0.05)
    nq = sum(1 for f in feats if q[fam][f] < 0.05)
    print(f"  {fam:18} raw_sig={nraw:2}  q_sig={nq:2}")
