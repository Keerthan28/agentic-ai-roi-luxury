# Data Access

## Summary

This directory holds the datasets that feed the modeling pipeline. Two raw datasets are involved:

| File                                                             | Role                | Shareable? |
| ---------------------------------------------------------------- | ------------------- | ---------- |
| `luxury_goods_synthetic_only_ai_positive.xlsx`                   | Synthetic training  | Yes        |
| `Luxury Goods Shopping Experince_March 11, 2026_17.48.xlsx`      | Real survey         | No (raw)   |

## 1. Synthetic Training Data

A 300-row synthetic dataset, biased toward AI-positive responses, used to fit the three ensemble regressors. This dataset is produced by the project team for model-training purposes only; it does **not** represent the real population and should not be used for inference about real consumer behavior on its own.

This file is safe to commit and redistribute. It is included in the repository for reproducibility.

## 2. Real Survey Data

A 43-row Qualtrics export from the "Luxury Goods Shopping Experience" survey conducted on March 11, 2026. Respondents were anonymous; no direct identifiers are present in the export. Despite the absence of direct identifiers, the project team has elected not to commit the raw file to the public repository to avoid incidental re-identification risk given the small sample size and because a formal public-release review has not been completed.

**What is shareable:**
- Aggregate statistics (means, medians, distributions, segment counts)
- Visualizations built from the real data (already in `results/`)
- Derived feature values at aggregate level (see `results/roi_model_results.csv` for the synthetic run)

**What is not shareable:**
- Per-row real survey responses
- Free-text open-ended answers

## 3. If You Need the Real Data

Contact the project lead (see README.md). Access will be granted after a brief data-use discussion. Typical use cases that warrant access:
- Replication of the transfer-learning evaluation
- Additional sensitivity analysis requested by the sponsor
- Extension studies building on the same instrument

## 4. Re-Running Without Real Data

The synthetic-only pipeline (`src/agentic_ai_roi_model.py`) runs end-to-end with only the synthetic file present and reproduces all synthetic-path tables and figures. The real-data script (`src/run_model_on_real_data.py`) requires the real file to run.
