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
| `openai` | Not specified in the reference README | API calls to OpenAI-compatible and Azure OpenAI endpoints |
| `httpx` | Not specified in the reference README | API timeout and HTTP configuration |
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

Run scripts from the repository root. For example:

```bash
python src/<script_name>.py
```

Some scripts require API access. Set the corresponding environment variables before running model-calling or annotation scripts:

```bash
export OPENAI_API_KEY="your_api_key"
export AZURE_OPENAI_API_KEY="your_azure_api_key"
export AZURE_OPENAI_ENDPOINT="your_azure_endpoint"
```

The exact variables required may differ by script depending on whether it uses Azure OpenAI, OpenRouter, SiliconFlow, or another OpenAI-compatible endpoint.

## Notes

- Large raw datasets and generated result files should be kept in `src/dataset/` and `results/` rather than mixed with core code.
- Do not commit API keys, passwords, local environment files, or private credentials.
- If results are regenerated, place them in the corresponding subfolder under `results/` so that each experiment remains easy to locate.

