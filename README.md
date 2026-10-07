# Heineken Account Risk Challenge

A reproducible prototype for account-level next-month risk scoring and rep prioritisation. This is a course-challenge demonstration using an adapted, anonymised Brazilian marketplace dataset; it is not HEINEKEN sales data.

## Project layout

- `src/`: maintained Python feature-building and exploratory-analysis scripts.
- `notebooks/`: exploratory analysis, XGBoost model training/evaluation, and risk-landscape visualisation.
- `data/raw/`: locally supplied source files. The dataset is not tracked in this repository; see `data/raw/README.md`.
- `outputs/`: generated rolling snapshots and score exports. Account-level files are excluded from Git by default.
- `docs/`: model approach and selected-feature dictionary.
- `presentation/`: project presentation material.
- `archive/`: superseded plots, preliminary outputs, source download archive, and assignment references retained locally.

## Setup

Use Python 3.13 or a compatible environment, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On macOS, XGBoost requires an OpenMP runtime. If XGBoost cannot load its native library, install `libomp` with a system package manager and restart the notebook kernel.

## Data setup

Obtain the challenge dataset through the course's authorized distribution. Extract the files so they are available at:

```text
data/raw/drive-download-20261006T150120Z-1-001/order_lines.csv
data/raw/drive-download-20261006T150120Z-1-001/order_reviews.csv
data/raw/drive-download-20261006T150120Z-1-001/geolocation.csv
```

The source archive and raw data are intentionally excluded from public Git history. Do not republish them unless their redistribution rights have been confirmed.

## Run order

1. From the repository root, run `python src/build_rolling_risk_dataset.py` to regenerate rolling account-month features, labels, and latest-snapshot eligibility status in `outputs/`.
2. Open and run `notebooks/train_high_risk_classifier.ipynb`. It evaluates the three binary risk classifiers on a chronological holdout, then refits separate scoring models on eligible labelled history and exports account-level scores, priority, and model explanations.
3. Open and run `notebooks/risk_scoring_landscape.ipynb` for descriptive plots from the score exports.
4. `notebooks/data_exploration.ipynb` contains earlier supporting exploration; it is not required for the final scoring pipeline.

Run the scripts from the project root so relative paths and output locations resolve as documented.

## Main outputs

- `outputs/current_account_priority_queue.csv`: compact rep-facing ranking with eligibility status, high-risk/churn/loss probabilities, revenue importance percentile, priority score, rank, and leading risk drivers.
- `outputs/current_account_risk_scores.csv`: wider account-level score and feature table.
- `outputs/current_account_feature_contributions.csv`: all selected feature values and signed model contributions for eligible accounts.
- `outputs/high_risk_test_predictions.csv`: temporal holdout predictions and actual labels.
- `outputs/high_risk_feature_importance.csv`: global model importance summary.
- `outputs/rolling_account_risk_dataset.csv`: point-in-time rolling features and rule-defined labels.
- `docs/MODEL_APPROACH.md`: detailed business and technical description, evaluation, data lineage, and front-end guidance.

Model probabilities describe the next eligible monthly snapshot. Accounts with fewer than three orders in the preceding 18 months are marked `insufficient_data` and are not assigned probabilities or a priority rank. The supplied data ends on 31 August 2018, so the included scores are historical demonstration outputs, not live account scores.

## Public-repository caution

Before publishing, confirm redistribution rights for challenge materials and review any new files for account-level or other sensitive data. `.gitignore` excludes raw data, account-level generated outputs, archive/reference material, the local virtual environment, and editor state by default. Only source code, notebooks, documentation, and approved aggregate artifacts should be committed.
