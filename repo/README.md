# Agentic AI ROI — Luxury Goods Industry

**Sponsor:** Deloitte Consulting
**Engagement:** Capstone — Final deliverable, Week 14
**Contact:** Keerthan Nekkanti — nekkanti.keerthan2826@gmail.com

---

## 1. Purpose

This repository contains the full pipeline, analysis, and supporting artifacts for a capstone project that quantifies the ROI of deploying agentic AI in the luxury goods retail industry. The work combines primary survey evidence from luxury shoppers with open-source retention benchmarks (Bain, McKinsey, Deloitte Global Powers of Luxury Goods, BCG x Altagamma) to produce a customer-level ROI model validated with three supervised-learning ensembles (XGBoost, Random Forest, Gradient Boosting).

The headline finding is an average projected net ROI of **1.11x per customer** on the cleaned modeling population, driven primarily by the intersection of AI readiness and customer lifetime value. See `docs/Final_Report.pdf` for the full narrative and recommendations.

## 2. Repository Layout

```
.
├── README.md                    <- this file
├── requirements.txt             <- pinned Python dependencies
├── LICENSE                      <- MIT license
├── .gitignore
├── data/
│   └── data-access.md           <- what is and is not shareable
├── src/
│   ├── agentic_ai_roi_model.py     <- end-to-end synthetic training pipeline
│   └── run_model_on_real_data.py   <- transfer + retrain evaluation on real survey
├── notebooks/
│   └── 01_exploration.ipynb     <- narrative walk-through (optional)
├── docs/
│   ├── data_dictionary.md       <- every feature documented
│   ├── methodology.md           <- modeling decisions, OSINT assumptions
│   ├── survey_instrument.md     <- Q1–Q23 question text and scales
│   └── ip_note.md               <- IP / license / data-sharing note
├── results/
│   ├── roi_model_results.csv    <- per-customer predictions
│   ├── agentic_ai_roi_analysis.png
│   └── real_data_roi_analysis.png
├── tests/
│   └── test_pipeline.py         <- basic feature-pipeline sanity checks
└── third_party/
    └── NOTICE.md                <- third-party library notices
```

## 3. Quick Start

```bash
# 1. Clone and enter
git clone <repo-url> agentic-ai-luxury-roi
cd agentic-ai-luxury-roi

# 2. Create an isolated environment (Python 3.10+)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Place the raw survey files in data/
#    (see data/data-access.md for what is available)

# 4. Run the synthetic training pipeline
python src/agentic_ai_roi_model.py

# 5. Run the real-data evaluation
python src/run_model_on_real_data.py
```

Both scripts write their outputs into the current working directory — PNGs for visualizations and a CSV for per-customer predictions.

## 4. What's Included vs. Held Out

| Artifact                              | In repo | Notes                                        |
| ------------------------------------- | ------- | -------------------------------------------- |
| Model code (synthetic + real)         | yes     | `src/*.py`                                   |
| Synthetic training data               | yes     | `data/luxury_goods_synthetic_only_ai_positive.xlsx` — AI-positive biased by design |
| Real survey data                      | no      | Aggregated results only; see `data/data-access.md` |
| OSINT benchmark table                 | yes     | `docs/methodology.md` Section 3              |
| Trained model artifacts (pickles)     | yes     | `results/` — regenerated on every run        |
| Final sponsor report                  | yes     | `docs/Final_Report.docx`                     |

## 5. Reproducibility

- Python 3.10+, all dependencies pinned in `requirements.txt`.
- Random seeds fixed at 42 across sklearn, xgboost, and numpy.
- Both scripts are deterministic end-to-end given the same input data.
- Run `pytest tests/` to execute the basic feature-pipeline sanity checks.

## 6. Key Results

| Metric                         | Value           |
| ------------------------------ | --------------- |
| Cleaned sample size            | 253             |
| Mean Net ROI                   | 1.11x           |
| Median Net ROI                 | 0.82x           |
| Mean AI Readiness              | 0.44            |
| Mean Retention Improvement     | +5.0 pts        |
| Mean CLV per customer          | $78,413         |
| Mean AI revenue uplift / cust. | $6,179          |
| Mean retention savings / cust. | $5,089          |
| Mean AI cost / cust.           | $5,328          |

See `results/roi_model_results.csv` for per-customer predictions and `docs/Final_Report.docx` for full discussion, segment analysis, and recommendations.

## 7. License

Released under the MIT license — see `LICENSE`. Third-party library notices in `third_party/NOTICE.md`.
No non-disclosure agreement is in effect for this engagement; however, raw survey data is not redistributed. See `docs/ip_note.md` for details.

## 8. Contact

Primary contact: **Keerthan Nekkanti** — nekkanti.keerthan2826@gmail.com
Sponsor: Deloitte Consulting
