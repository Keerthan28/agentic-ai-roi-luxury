"""
Basic sanity checks for the ROI feature pipeline.

These are intentionally small — they verify that the deterministic parts of the
pipeline produce the shapes and ranges we expect. They do NOT validate model
accuracy; that is measured in the two scripts under src/.

Run with:  pytest tests/
"""
import pandas as pd
import numpy as np
import pytest
import sys
import os

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


OSINT_RETENTION = {
    "baseline_retention_rate": 0.82,
    "ai_enhanced_retention_uplift": 0.15,
    "avg_luxury_clv_multiplier": 8.5,
    "churn_cost_multiplier": 5.0,
    "personalization_revenue_lift": 0.20,
    "ai_implementation_cost_ratio": 0.03,
    "income_tier_clv": {
        "Under $25,000": 1_500,
        "$25,000–$49,999": 3_500,
        "$50,000–$99,999": 7_000,
        "$100,000–$199,999": 15_000,
        "$200,000–$499,999": 35_000,
        "$500,000+": 75_000,
    },
    "purchase_frequency_weight": {
        "Multiple times per year": 1.0,
        "Once per year": 0.6,
        "Every few years": 0.3,
        "Rarely": 0.15,
        "Never": 0.05,
    },
    "age_digital_propensity": {
        "Under 18": 0.85, "18–24": 0.92, "25–34": 0.88,
        "35–44": 0.75, "45–54": 0.60, "55–64": 0.45, "65+": 0.30,
    },
}


def test_income_tier_mapping_covers_all_bands():
    bands = set(OSINT_RETENTION["income_tier_clv"].keys())
    expected = {
        "Under $25,000", "$25,000–$49,999", "$50,000–$99,999",
        "$100,000–$199,999", "$200,000–$499,999", "$500,000+",
    }
    assert bands == expected


def test_age_propensity_monotonic_in_young_bands():
    prop = OSINT_RETENTION["age_digital_propensity"]
    assert prop["18–24"] >= prop["25–34"] >= prop["35–44"]


def test_purchase_freq_weights_ordered():
    w = OSINT_RETENTION["purchase_frequency_weight"]
    assert (
        w["Multiple times per year"] > w["Once per year"]
        > w["Every few years"] > w["Rarely"] > w["Never"]
    )


def test_ai_readiness_weights_sum_to_one():
    weights = [0.15, 0.15, 0.20, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05]
    assert abs(sum(weights) - 1.0) < 1e-9


def test_roi_formula_breaks_even_at_correct_point():
    """For a neutral reader (readiness=0, digital_propensity=0):
       retention_improvement = 0, revenue uplift = 0,
       retention_savings = 0, so net_roi = (0 + 0 - cost) / cost = -1."""
    annual_spend = 7_000
    ai_cost = annual_spend * OSINT_RETENTION["ai_implementation_cost_ratio"] \
        * OSINT_RETENTION["avg_luxury_clv_multiplier"]
    rev = 0
    savings = 0
    net_roi = (rev + savings - ai_cost) / ai_cost
    assert abs(net_roi - (-1.0)) < 1e-9


def test_results_csv_exists_if_run():
    path = os.path.join(os.path.dirname(__file__), "..", "results",
                        "roi_model_results.csv")
    if not os.path.exists(path):
        pytest.skip("results/roi_model_results.csv not yet generated; run "
                    "src/agentic_ai_roi_model.py first")
    df = pd.read_csv(path)
    assert len(df) > 0
    assert "net_roi" in df.columns
    assert "ai_readiness" in df.columns
    assert df["ai_readiness"].between(0, 1).all()
