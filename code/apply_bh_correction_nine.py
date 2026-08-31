"""Benjamini-Hochberg FDR correction on the nine mechanism-feature regressions.

Per the rebuttal (Reviewer UTZC, S1): "applied Benjamini-Hochberg FDR correction
to all textual feature regressions, treating each regression as a separate family
of nine tests." Here each of the three regressions is one family of 9 tests:
  - Human           (human_delta)
  - LLM-first-person (agent_delta, pooled + model FE)
  - LLM-observer     (delta_awarded, pooled + model FE)

For each family the 9 raw p-values are corrected with statsmodels
multipletests(method="fdr_bh") to obtain q-values. Uses the corrected
nine-feature regressions recomputed from raw (H1 API-failure branches excluded;
all 8 models incl. gpt-5.5). Prints a p/q table to stdout.
"""
import os
import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REC = os.path.join(ROOT, "regression_output")

# internal paper_feature_name -> display label
LABELS = {
    "OP Length": "OP Length", "OP Def Freq": "OP Definitive", "OP I Freq": "OP 1stPerson",
    "Ch We Freq": "Reply Inclusive", "Ch Has Formatting": "Reply Formatting",
    "Ch Def Freq": "Reply Definitive", "Ch Dissimilarity": "Reply Dissimilarity",
    "Ch Has Link": "Reply Link", "Ch Length": "Reply Length",
}
KEYS = list(LABELS)
FAMILIES = {
    "Human": ("human_nine/human_logistic_regression_coefficients_all.csv", False),
    "LLM-first-person": ("agent_nine/agent_logistic_regression_coefficients_all.csv", True),
    "LLM-observer": ("observer_nine/agent_logistic_regression_coefficients_all.csv", True),
}


def load_p(path, pooled):
    d = pd.read_csv(path)
    d = d[d["term_type"] == "feature"]
    if pooled:
        d = d[d["regression"] == "pooled_model_fixed_effects"]
    return d.set_index("paper_feature_name")["p_value"]


# apply BH per family (9 tests each)
res = {}
for fam, (rel, pooled) in FAMILIES.items():
    p = load_p(os.path.join(REC, rel), pooled)
    pv = [float(p[k]) for k in KEYS]
    qv = multipletests(pv, method="fdr_bh")[1]
    res[fam] = dict(zip(KEYS, zip(pv, qv)))

# long table
rows = []
for k in KEYS:
    row = {"Feature": LABELS[k]}
    for fam in FAMILIES:
        pv, qv = res[fam][k]
        row[f"{fam}_p"] = round(pv, 4)
        row[f"{fam}_q"] = round(qv, 4)
        row[f"{fam}_sig(q<.05)"] = "yes" if qv < 0.05 else "no"
    rows.append(row)
df = pd.DataFrame(rows)

print(df.to_string(index=False))
