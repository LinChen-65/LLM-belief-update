# Do LLMs Change Their Minds Like Humans? Diagnosing Human–LLM Divergence in Single-Turn Persuasion Judgments

Code and data for the paper. The project studies whether large language models make belief-update judgments in a human-like way, using Reddit ChangeMyView-style paired replies with participant-verified persuasion outcomes.

## Layout

```
code/    all scripts (data generation, regression, analysis, visualization)
data/    all raw data
  dataset/         input corpus and lexicon (OP metadata, paired replies, NRC-VAD lexicon)
  first_person/    first-person judgment outputs (model role-plays as the original poster)
  third_person/    observer judgment outputs (model predicts the OP's view change)
  justification/   GPT-5.1 continuous belief-change scores (0–100) of each justification
  random_seed/     three repeated first-person runs per model (stability analysis)
  strategy/        persuasion-strategy annotations of replies
  topic/           proposition-type annotations of original posts
```

Scripts resolve paths relative to the repository root, so run them from the repo root. Figures are written to a `figures/` directory and regression coefficient tables to a `regression_output/` directory; both are created at run time and are not part of the tracked repository.

## Requirements

Python 3. Core packages:

```bash
pip install openai httpx numpy pandas matplotlib scikit-learn scipy statsmodels seaborn
```

API-calling scripts read credentials from environment variables (e.g. `SF_API_KEY` for SiliconFlow, `AZURE_OPENAI_API_KEY` for Azure OpenAI). No keys are stored in the code.

## Data generation (optional; outputs already provided under `data/`)

| Script | Produces |
|---|---|
| `agent_first_person_judgment.py` | First-person belief-update judgments → `data/first_person/` |
| `agent_third_person_judgment.py` | Observer belief-update judgments → `data/third_person/` |
| `topic_classification_azure.py` | Proposition-type annotations → `data/topic/` |
| `strategy_classification_azure.py` | Persuasion-strategy annotations → `data/strategy/` |

## Reproducing the figures and tables

Most figure scripts read `data/` directly and can be run in one step:

```bash
python3 code/<script>.py        # writes figures/<name>.png
```

Four items depend on regression coefficient tables, so first run the relevant regression script(s) (which write to `regression_output/`), then the plotting/analysis script. Always pass the VAD lexicon:

```bash
VAD=data/dataset/NRC-VAD-Lexicon-v2.1.txt
python3 code/agent_persuasion_logistic_regression_nine_mechanism_features.py --vad_lexicon $VAD
```

### Figure → code map

| Figure / Table | Script(s) to run |
|---|---|
| Fig. 1 (first-person error composition + κ) | `plot_anomalies_and_kappa.py` |
| Fig. 2 (nine-feature logistic OR, human vs LLM) | `human_persuasion_logistic_regression_nine_mechanism_features.py`, `agent_persuasion_logistic_regression_nine_mechanism_features.py` → `plot_regression_textual_features.py` |
| Fig. 3 (agreement/error by proposition type) | `plot_proposition_type_agreement_error.py` |
| Fig. 4 (persuasion-strategy distribution) | `plot_distribution_strategy.py` |
| Fig. 5 (strategy effectiveness, human vs LLM) | `plot_strategy_effectiveness.py` |
| Fig. 6 (first-person vs observer FN/FP) | `plot_perspective_comparison.py` |
| Fig. 7 (feature-coefficient change, first-person→observer) | `{human,agent,observer}_persuasion_logistic_regression_nine_mechanism_features.py` → `plot_regression_change.py` |
| Fig. 8 (first-person vs observer by proposition type) | `plot_perspective_comparison_by_proposition_type.py` |
| Fig. 9 (first-person vs observer by strategy) | `plot_strategy_effectiveness_first_person_observer.py` |
| Fig. 14 (inter-model κ) | `plot_model_kappa.py` |
| Fig. 15 (OLS on continuous belief-change score) | `agent_persuasion_logistic_regression_nine_mechanism_features.py` → `plot_belief_change_score.py` |
| Fig. 16 (observer error composition + κ) | `plot_discrepancy_human_llmobserver.py` |
| Fig. 17 (first-person vs observer consistency) | `plot_perspective_consistency.py` |
| Table 1 (judgment stability across repeated runs) | `compute_stability.py` |
| Table 2 (full 40 textual features + BH-FDR) | `{human,agent,observer}_persuasion_logistic_regression.py` → `apply_bh_correction_table2.py` |

Figs. 10–13 are prompt templates (shown from the judgment/annotation scripts) and have no plotting code.
