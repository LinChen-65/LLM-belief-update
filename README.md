# Code for paper: Do LLMs Change Their Minds Like Humans? Diagnosing Human–LLM Divergence in Belief Update

This repository contains the core experimental code, datasets, and processed results for the manuscript **Do LLMs Change Their Minds Like Humans? Diagnosing Human–LLM Divergence in Belief Update**.

The project studies whether large language models make belief-update judgments in a human-like way. Using Reddit ChangeMyView-style paired replies, the experiments compare human delta labels with LLM judgments under different evaluation perspectives and diagnostic conditions.

## Layout

- `src/`: core code for running experiments and analyses.
  - `src/dataset/`: datasets required by the experiments.
  - Other `src/*.py` files: core scripts for model calls, annotation, evaluation, analysis, and plotting.
  - Additional `src/*.py` files: robustness checks and validation utilities added for the revised analysis.
- `results/`: core experimental results reported in the manuscript.
  - `results/first_person/`: results for the first-person belief-update judgment experiment, where the model acts as the original poster.
  - `results/third_person/`: results for the third-person observer judgment experiment, where the model predicts whether the original poster would change their view.
  - `results/topic/`: results for the topic-dimension analysis, grouping original posts by proposition type.
  - `results/strategy/`: results for the persuasion-strategy analysis, grouping challenger replies by argument strategy.
  - `results/topic_model_agreement/`: cross-model validation results for proposition-type annotation agreement between GPT-5.1 and GLM-5.2.
  - `results/strategy_effectiveness_plot/`: reply-level and model-level persuasion-strategy effectiveness summaries and figures.
  - `results/multiple_comparison_correction_table1/`: Benjamini-Hochberg correction outputs for the textual-feature results reported in Appendix Table 1.
  - `results/sampling_robustness/`: repeated-run consistency diagnostics and mismatch records for sampling-robustness analysis.

## Requirements

The code is written in Python. The main scripts use the following packages:

| Package | Version | Purpose |
| --- | --- | --- |
| `openai` | `2.14.0` | API calls to OpenAI-compatible and Azure OpenAI endpoints |
| `httpx` | `0.28.1` | API timeout and HTTP configuration |
| `numpy` | `2.2.6` | numerical computation |
| `pandas` | `2.2.3` | data processing and tabulation |
| `matplotlib` | `3.10.5` | plotting figures |
| `seaborn` | compatible with the installed Matplotlib version | statistical visualization for topic and strategy analyses |
| `scikit-learn` | `1.7.1` | metrics such as Cohen's kappa |
| `scipy` | `1.15.3` | statistical analysis |
| `statsmodels` | `0.14.5` | regression analysis |

install the core packages with:

```bash
pip install openai httpx numpy pandas matplotlib scikit-learn scipy statsmodels
pip install seaborn
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
| Topic annotation agreement | GPT-5.1 and GLM-5.2 proposition-type annotations are aligned over the full corpus and compared using raw agreement and Cohen's kappa. | processed validation pipeline | `results/topic_model_agreement/` |
| Sampling robustness | Repeated first-person runs are compared using exact-match rate, pairwise agreement, pairwise Cohen's kappa, and Fleiss' kappa. | `src/check_gemini_agent_delta_consistency.py` | `results/sampling_robustness/` |
| Multiple-comparison correction | Benjamini-Hochberg correction is applied separately to the human, first-person LLM, and observer LLM textual-feature regression families reported in Appendix Table 1. | processed statistical analysis | `results/multiple_comparison_correction_table1/` |
| Strategy effectiveness visualization | Human and model persuasion rates are summarized by logos, pathos, ethos, and their combinations. | `src/plot_persuasion_strategy_human_vs_first_person.py` | `results/strategy_effectiveness_plot/` |

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

### Robustness and validation scripts

| Script | Purpose |
| --- | --- |
| `check_gemini_agent_delta_consistency.py` | Compares three repeated Gemini first-person runs at the reply-branch level. It reports exact three-run agreement, pairwise agreement, pairwise Cohen's kappa, and Fleiss' kappa, and saves cases whose `agent_delta` labels differ across runs. |

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

## Additional Validation and Result Files

### Topic model agreement

The `results/topic_model_agreement/` directory contains the full-corpus proposition-type comparison between GPT-5.1 and GLM-5.2. The two annotation sets cover the same 2,262 original posts and achieve Cohen's kappa of approximately 0.751.

| File | Description |
| --- | --- |
| `topic_model_agreement_aligned_detail.csv` | Reply-independent OP-level records with GPT-5.1 and GLM-5.2 labels, agreement indicators, and original text. |
| `topic_model_agreement_summary.json` | Corpus coverage, raw agreement, Cohen's kappa, label distributions, and per-class statistics. |
| `topic_model_agreement_confusion_matrix.csv` | Confusion matrix for fact, value, and policy labels. |
| `topic_model_agreement_label_distribution.csv` | Label counts and proportions for both annotation models. |
| `topic_model_agreement_mismatches.csv` | Cases where the two models assign different proposition types. |
| `agreement_summary_bar.svg` | Visual summary of raw agreement and disagreement. |
| `confusion_matrix_heatmap.svg` | Heatmap of the cross-model confusion matrix. |
| `label_distribution_bar.svg` | Comparison of label distributions across the two models. |

### Sampling robustness

The repeated-run consistency analysis treats the main Gemini result file and two additional seeded result files as three independent annotation runs. The analysis is performed separately for the human-success and human-failure branches and for both branches combined.

| File | Description |
| --- | --- |
| `gemini_agent_delta_mismatches.csv` | Cases where at least one of the three Gemini runs produces a different `agent_delta` label. |

### Persuasion-strategy effectiveness

The `results/strategy_effectiveness_plot/` directory contains the processed reply-level records and summaries used to compare human persuasion rates with first-person model persuasion rates across strategy combinations.

| File | Description |
| --- | --- |
| `reply_level_strategy_results_long.csv` | Long-format reply-level data containing pair ID, branch, strategy combination, human label, model prediction, and model name. |
| `human_strategy_summary.csv` | Human persuasion rate and sample size for each strategy combination. |
| `model_strategy_summary.csv` | Model-specific persuasion rate by strategy combination. |
| `mean_strategy_summary.csv` | Across-model mean persuasion rate compared with the human rate. |
| `persuasion_strategy_effectiveness_scatter.png` | Raster version of the strategy-effectiveness scatter plot. |
| `persuasion_strategy_effectiveness_scatter.pdf` | Vector-ready publication version of the same plot. |

### Multiple-comparison correction

The `results/multiple_comparison_correction_table1/` directory contains Benjamini-Hochberg adjusted results for the 40 textual features included in Appendix Table 1. The human, first-person LLM, and observer LLM regressions are treated as three separate testing families.

| File | Description |
| --- | --- |
| `table1_exact_results_normalized.csv` | Normalized regression coefficients and raw p-values collected from the three source regressions. |
| `table1_bh_corrected_long.csv` | Long-format table containing raw p-values, BH-adjusted q-values, and significance indicators. |
| `table1_bh_corrected_wide_for_manuscript.csv` | Wide-format table prepared for manuscript reporting. |
| `table1_bh_significance_summary.csv` | Number of significant features before and after correction for each testing family. |
| `table1_significance_changed_after_bh.csv` | Features whose significance status changes after BH correction. |
| `table1_excluded_terms_or_unmapped_features.csv` | Regression terms excluded from correction or not mapped to manuscript feature names. |
| `table1_multiple_comparison_correction_summary.json` | Analysis configuration, input metadata, output paths, and correction summary. |
| `plots/table1_significance_counts_before_after_bh.svg` | Comparison of significant-feature counts before and after correction. |
| `plots/table1_bh_q_value_heatmap.svg` | Heatmap of BH-adjusted q-values. |
| `plots/table1_raw_p_vs_bh_q.svg` | Comparison between raw p-values and adjusted q-values. |

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

# 11. Check repeated-run consistency for Gemini
python src/check_gemini_agent_delta_consistency.py
```

Before running scripts that call APIs, set the required credentials, for example:

```bash
export SF_API_KEY="your_siliconflow_api_key"
export OPENROUTER_API_KEY="your_openrouter_api_key"
export AZURE_OPENAI_API_KEY="your_azure_api_key"
export AZURE_OPENAI_ENDPOINT="your_azure_endpoint"
```

The exact environment variables required may differ by script depending on whether it uses SiliconFlow, OpenRouter, Azure OpenAI, or another OpenAI-compatible endpoint.
