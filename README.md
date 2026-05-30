# Code for paper: Do LLMs Change Their Minds Like Humans? Diagnosing Human–LLM Divergence in Belief Update

This repository contains the core experimental code, datasets, and processed results for the manuscript **Do LLMs Change Their Minds Like Humans? Diagnosing Human–LLM Divergence in Belief Update**.

The project studies whether large language models make belief-update judgments in a human-like way. Using Reddit ChangeMyView-style paired replies, the experiments compare human delta labels with LLM judgments under different evaluation perspectives and diagnostic conditions.

## Layout

- `src/`: core code for running experiments and analyses.
  - `src/dataset/`: datasets required by the experiments.
  - Other `src/*.py` files: core scripts for model calls, annotation, evaluation, analysis, and plotting.
- `results/`: core experimental results reported in the manuscript.
  - `results/first_person/`: results for the first-person belief-update judgment experiment, where the model acts as the original poster.
  - `results/third_person/`: results for the third-person observer judgment experiment, where the model predicts whether the original poster would change their view.
  - `results/topic/`: results for the topic-dimension analysis, grouping original posts by proposition type.
  - `results/strategy/`: results for the persuasion-strategy analysis, grouping challenger replies by argument strategy.

## Requirements

The code is written in Python. The main scripts use the following packages:

| Package | Version | Purpose |
| --- | --- | --- |
| `openai` | 2.14.0 | API calls to OpenAI-compatible and Azure OpenAI endpoints |
| `httpx` | 0.28.1 | API timeout and HTTP configuration |
| `numpy` | `2.2.6` | numerical computation |
| `pandas` | `2.2.3` | data processing and tabulation |
| `matplotlib` | `3.10.5` | plotting figures |
| `scikit-learn` | `1.7.1` | metrics such as Cohen's kappa |
| `scipy` | `1.15.3` | statistical analysis |
| `statsmodels` | `0.14.5` | regression analysis |

install the core packages with:

```bash
pip install openai httpx numpy pandas matplotlib scikit-learn scipy statsmodels
```

## Data

The experimental datasets are stored under:

```text
src/dataset/
```

Typical data files include paired ChangeMyView replies, original post metadata, and annotated variants used for topic and strategy analyses.

## Main Experiments and Outputs

| Experiment | Description | Code location | Result location |
| --- | --- | --- | --- |
| First-person judgment | LLM acts as the original poster and decides whether the challenger changed its view. | `src/` | `results/first_person/` |
| Third-person judgment | LLM acts as an external observer and predicts whether the challenger changed the OP's view. | `src/` | `results/third_person/` |
| Topic analysis | Original posts are classified into fact, value, and policy propositions; bias is compared within each topic type. | `src/` | `results/topic/` |
| Strategy analysis | Challenger replies are classified by persuasion strategy; human and LLM persuasion rates are compared across strategy types. | `src/` | `results/strategy/` |

## Reproducing the Analyses

Run scripts from the repository root. Some scripts require API access and should be run only after setting the corresponding environment variables for SiliconFlow, OpenRouter, Azure OpenAI, or other OpenAI-compatible endpoints.

### Experiment scripts

| Script | Purpose |
| --- | --- |
| `agent_first_person_judgment.py` | Runs the first-person belief-update judgment experiment. The model acts as the original poster and decides whether each challenger reply changes its own view. The script supports multiple model providers, including SiliconFlow, OpenRouter, and Azure OpenAI. |
| `agent_third_person_judgment.py` | Runs the third-person observer judgment experiment. The model acts as an impartial external observer and predicts whether each challenger reply would change the original poster's view. |

### Annotation and data-preparation scripts

| Script | Purpose |
| --- | --- |
| `topic_classification_azure.py` | Classifies original posts into proposition types: fact, value, or policy. It uses Azure OpenAI with a structured JSON output format and saves topic labels for later topic-bias analysis. |
| `strategy_classification_azure.py` | Annotates challenger replies with persuasion-strategy labels: logos, pathos, and ethos. Each reply can receive multiple labels, and the output is saved back into the paired-reply data structure. |
| `retry_failed_topics.py` | Repairs failed topic-classification cases. It scans the topic classification result file for entries marked as `ERROR_FAILED`, reloads the original post text, and re-runs topic classification only for failed items. |
| `retry_failed_strategies.py` | Repairs failed or incomplete persuasion-strategy annotations. It identifies replies with missing or invalid `logos`, `pathos`, and `ethos` labels, then re-calls Azure OpenAI to re-annotate those failed cases. |
| `rewrite_error_ops.py` | Rewrites original posts that may trigger Azure content filters. It uses DeepSeek through SiliconFlow to produce sanitized versions of OP titles and bodies while preserving the original argument and intent. |
| `rewrite_replies.py` | Rewrites challenger replies that may trigger Azure content filters or fail strategy annotation. It sanitizes sensitive wording while preserving the reply's argument, reasoning, and persuasion strategy. |

### Regression analysis scripts

| Script | Purpose |
| --- | --- |
| `human_persuasion_logistic_regression.py` | Extracts a broad set of textual and interaction features from original posts and challenger replies, then fits logistic regression models using human delta labels as the outcome. |
| `agent_persuasion_logistic_regression.py` | Applies the broad text-feature logistic regression framework to first-person LLM judgments. |
| `observer_persuasion_logistic_regression.py` | Applies the broad text-feature logistic regression framework to third-person observer judgments. |
| `human_persuasion_logistic_regression_nine_mechanism_features.py` | Fits a reduced logistic regression model for human labels using nine mechanism-oriented features. |
| `agent_persuasion_logistic_regression_nine_mechanism_features.py` | Fits the same nine-feature mechanism model using first-person LLM judgments as the outcome. |
| `observer_persuasion_logistic_regression_nine_mechanism_features.py` | Fits the same nine-feature mechanism model using third-person observer judgments as the outcome. |

### Topic and strategy analysis scripts

| Script | Purpose |
| --- | --- |
| `topic_bias_analysis.py` | Combines topic labels with model judgment results, groups cases by proposition type, and computes bias metrics such as Cohen's kappa, FN rate, and FP rate for each topic category. |
| `strategy_analysis.py` | Combines persuasion-strategy labels with observer model judgments, groups replies by strategy combination, and computes alignment and bias metrics such as success-rate gaps, Cohen's kappa, FN rate, FP rate, and significance tests. |

### Plotting and figure scripts

| Script | Purpose |
| --- | --- |
| `plot_all_models_fp_fn_kappa.py` | Loads model judgment results, computes FP, FN, and Cohen's kappa for each model, and plots a stacked FP/FN bar chart with a kappa line. |
| `plot_all_models_fp_fn_kappa_stacked_bar.py` | Produces a publication-style stacked bar chart using manually specified FP, FN, total error, and kappa values across models. |
| `plot_observer_all_models_fp_fn_kappa.py` | Computes and visualizes FP, FN, and Cohen's kappa for the third-person observer condition across all models. |
| `plot_perspective_fn_fp_comparison_dumbbell.py` | Creates a two-panel dumbbell plot comparing first-person and observer conditions for FN and FP rates across models. |
| `plot_pairwise_cohens_kappa_heatmap.py` | Plots a lower-triangular heatmap of pairwise Cohen's kappa agreement between models. |
| `plot_mechanism_feature_or_dumbbell.py` | Parses logistic regression summary files and plots a dumbbell-style odds-ratio comparison among first-person LLM, observer LLM, and human reference results for the nine mechanism features. |
| `plot_regression_comparison.py` | Parses human and first-person agent regression reports, compares odds ratios and confidence intervals, and performs coefficient-difference tests for same-direction effects. |
| `plot_three_way_regression_comparison.py` | Parses human, first-person agent, and third-person agent regression reports, then produces a three-way odds-ratio comparison plot. |
| `plot_strategy_distribution.py` | Analyzes the distribution of persuasion-strategy combinations in `single_turn_pairs_with_strategies.json` and plots a pie chart of strategy combinations. |
| `plot_persuasion_strategy_human_vs_first_person.py` | Plots human persuasion rates against first-person LLM persuasion rates by persuasion-strategy type. Individual model points and across-model means are shown. |
| `plot_strategy_effectiveness_first_vs_observer.py` | Computes persuasion rates by strategy combination for human labels, first-person LLM judgments, and observer judgments. It produces scatter plots comparing Human vs First-person, Human vs Observer, and First-person vs Observer. |

### Example workflow

```bash
# 1. Run model judgment experiments
python src/agent_first_person_judgment.py
python src/agent_third_person_judgment.py

# 2. Run topic and strategy annotation
python src/topic_classification_azure.py
python src/strategy_classification_azure.py

# 3. Repair failed annotations if needed
python src/retry_failed_topics.py
python src/retry_failed_strategies.py

# 4. Rewrite content-filtered cases if needed
python src/rewrite_error_ops.py
python src/rewrite_replies.py

# 5. Run feature-based regression analyses
python src/human_persuasion_logistic_regression.py
python src/agent_persuasion_logistic_regression.py
python src/observer_persuasion_logistic_regression.py

# 6. Run nine-mechanism-feature regression analyses
python src/human_persuasion_logistic_regression_nine_mechanism_features.py
python src/agent_persuasion_logistic_regression_nine_mechanism_features.py
python src/observer_persuasion_logistic_regression_nine_mechanism_features.py

# 7. Run topic and strategy analyses
python src/topic_bias_analysis.py
python src/strategy_analysis.py

# 8. Generate model-level judgment summary figures
python src/plot_all_models_fp_fn_kappa.py
python src/plot_all_models_fp_fn_kappa_stacked_bar.py
python src/plot_observer_all_models_fp_fn_kappa.py
python src/plot_perspective_fn_fp_comparison_dumbbell.py
python src/plot_pairwise_cohens_kappa_heatmap.py

# 9. Generate regression comparison figures
python src/plot_mechanism_feature_or_dumbbell.py
python src/plot_regression_comparison.py
python src/plot_three_way_regression_comparison.py

# 10. Generate strategy-analysis figures
python src/plot_strategy_distribution.py
python src/plot_persuasion_strategy_human_vs_first_person.py
python src/plot_strategy_effectiveness_first_vs_observer.py
```

Before running scripts that call APIs, set the required credentials, for example:

```bash
export SF_API_KEY="your_siliconflow_api_key"
export OPENROUTER_API_KEY="your_openrouter_api_key"
export AZURE_OPENAI_API_KEY="your_azure_api_key"
export AZURE_OPENAI_ENDPOINT="your_azure_endpoint"
```

The exact environment variables required may differ by script depending on whether it uses SiliconFlow, OpenRouter, Azure OpenAI, or another OpenAI-compatible endpoint.


