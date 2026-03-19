"""
Reusable feature engineering and projection pipeline for the Agentic AI ROI model.
All functions accept an `osint` parameter dict so the Streamlit app can override defaults.
"""

import pandas as pd
import numpy as np

# ── Default OSINT benchmarks ──────────────────────────────────────────────

DEFAULT_OSINT = {
    'baseline_retention_rate': 0.82,
    'ai_enhanced_retention_uplift': 0.15,
    'avg_luxury_clv_multiplier': 8.5,
    'churn_cost_multiplier': 5.0,
    'personalization_revenue_lift': 0.20,
    'ai_implementation_cost_ratio': 0.03,
    'income_tier_clv': {
        'Under $25,000': 1_500,
        '$25,000–$49,999': 3_500,
        '$50,000–$99,999': 7_000,
        '$100,000–$199,999': 15_000,
        '$200,000–$499,999': 35_000,
        '$500,000+': 75_000,
    },
    'purchase_frequency_weight': {
        'Multiple times per year': 1.0,
        'Once per year': 0.6,
        'Every few years': 0.3,
        'Rarely': 0.15,
        'Never': 0.05,
    },
    'age_digital_propensity': {
        'Under 18': 0.85, '18–24': 0.92, '25–34': 0.88,
        '35–44': 0.75, '45–54': 0.60, '55–64': 0.45, '65+': 0.30,
    },
}

# ── Valid response values (for cleaning Qualtrics exports) ────────────────

VALID_VALUES = {
    'Q1': {'Under 18', '18–24', '25–34', '35–44', '45–54', '55–64', '65+'},
    'Q2': {'Male', 'Female', 'Non-binary', 'Prefer not to say'},
    'Q3': {'Under $25,000', '$25,000–$49,999', '$50,000–$99,999',
            '$100,000–$199,999', '$200,000–$499,999', '$500,000+'},
    'Q4': {'Multiple times per year', 'Once per year', 'Every few years', 'Rarely', 'Never'},
    'Q7': {'Extremely dissatisfied', 'Somewhat dissatisfied',
            'Neither satisfied nor dissatisfied', 'Somewhat satisfied', 'Extremely satisfied'},
    'Q11': {'Not at all', 'A little', 'A moderate amount', 'A lot'},
    'Q13': {'Never', 'Rarely', 'Occasionally', 'Yes, frequently'},
    'Q16': {'yes', 'Yes', 'No', 'no'},
    'Q18': {'Never', 'Rarely', 'Occasionally', 'Frequently'},
    'Q19': {'N/A I have never used AI for luxury shopping', '1 Not Helpful',
             '2 Indifferent', '3 Somewhat Helpful', '4 Helpful'},
    'Q21': {'No, not at all', 'No, I prefer traditional luxury experiences',
             'Neutral', 'Yes, but in limited ways', 'Yes, definitely'},
    'Q22': {'I prefer no AI in luxury', 'Mostly human, minimal AI', 'Balanced mix of both'},
}

FEATURE_COLS = [
    'annual_spend', 'purchase_freq_weight', 'digital_propensity',
    'satisfaction_score', 'ai_usage_freq', 'ai_helpfulness',
    'ai_desire', 'ai_balance', 'auth_trust', 'ai_for_decision',
    'ai_use_case_count', 'ai_concern_count', 'desired_ai_roles',
    'brand_count', 'is_female', 'ai_assistant_freq', 'ai_readiness',
    'clv', 'retention_improvement', 'retention_adjusted_clv',
]

AI_USE_CASES = [
    'Product recommendations', 'Comparing prices',
    'Authenticating second-hand items', 'Styling advice',
    'Virtual try-on', 'Customer service chat assistants',
    'Researching brand history or product details',
]
CONCERN_ITEMS = ['Lack of personalization', 'Data privacy concerns',
                 'Inaccuracy', 'Removes human touch']
AI_ROLES = ['Personalization', 'Efficiency', 'Improve authenticity',
            'Assist staff', 'Create immersive digital experiences']
LUXURY_BRANDS = ['Louis Vuitton', 'Chanel', 'Gucci', 'Hermès', 'Dior',
                 'Prada', 'Rolex', 'Cartier']

ORDERED_SCALES = {
    'Q13': ['Never', 'Rarely', 'Occasionally', 'Yes, frequently'],
    'Q16': ['No', 'yes'],
    'Q18': ['Never', 'Rarely', 'Occasionally', 'Frequently'],
    'Q19': ['N/A I have never used AI for luxury shopping', '1 Not Helpful',
            '2 Indifferent', '3 Somewhat Helpful', '4 Helpful'],
    'Q21': ['No, not at all', 'No, I prefer traditional luxury experiences',
            'Neutral', 'Yes, but in limited ways', 'Yes, definitely'],
    'Q22': ['I prefer no AI in luxury', 'Mostly human, minimal AI', 'Balanced mix of both'],
}

DEFAULT_SHIFT_RATES = {
    'Q13': 0.28, 'Q16': 0.22, 'Q18': 0.25,
    'Q19': 0.20, 'Q21': 0.30, 'Q22': 0.25,
}


# ── Data cleaning ─────────────────────────────────────────────────────────

def clean_dataframe(df, drop_header_row=False):
    """Clean a raw Qualtrics export: coerce invalid values to NaN, keep valid age rows."""
    d = df.copy()
    if drop_header_row:
        d = d.iloc[1:].reset_index(drop=True)
    for col, valid in VALID_VALUES.items():
        if col in d.columns:
            d[col] = d[col].where(d[col].isin(valid))
    d = d[d['Q1'].notna()].reset_index(drop=True)
    return d


# ── Feature engineering (parameterized) ───────────────────────────────────

def engineer_features(df, osint=None):
    """Full feature engineering pipeline. Pass custom `osint` dict to override defaults."""
    R = osint or DEFAULT_OSINT
    d = df.copy()

    for col, valid in VALID_VALUES.items():
        if col in d.columns:
            d[col] = d[col].where(d[col].isin(valid))

    d['annual_spend'] = d['Q3'].map(R['income_tier_clv']).fillna(5_000)
    d['purchase_freq_weight'] = d['Q4'].map(R['purchase_frequency_weight']).fillna(0.3)
    d['digital_propensity'] = d['Q1'].map(R['age_digital_propensity']).fillna(0.5)

    d['satisfaction_score'] = d['Q7'].map({
        'Extremely dissatisfied': 1, 'Somewhat dissatisfied': 2,
        'Neither satisfied nor dissatisfied': 3, 'Somewhat satisfied': 4,
        'Extremely satisfied': 5,
    }).fillna(3)

    d['ai_usage_freq'] = d['Q13'].map({
        'Never': 0, 'Rarely': 1, 'Occasionally': 2, 'Yes, frequently': 3,
    }).fillna(0)

    d['ai_helpfulness'] = d['Q19'].map({
        'N/A I have never used AI for luxury shopping': 0,
        '1 Not Helpful': 1, '2 Indifferent': 2,
        '3 Somewhat Helpful': 3, '4 Helpful': 4,
    }).fillna(0)

    d['ai_desire'] = d['Q21'].map({
        'No, not at all': 0, 'No, I prefer traditional luxury experiences': 1,
        'Neutral': 2, 'Yes, but in limited ways': 3, 'Yes, definitely': 4,
    }).fillna(2)

    d['ai_balance'] = d['Q22'].map({
        'I prefer no AI in luxury': 0, 'Mostly human, minimal AI': 1,
        'Balanced mix of both': 2,
    }).fillna(1)

    d['auth_trust'] = d['Q11'].map({
        'Not at all': 0, 'A little': 1, 'A moderate amount': 2, 'A lot': 3,
    }).fillna(1)

    d['ai_for_decision'] = d['Q16'].str.lower().eq('yes').astype(int) if 'Q16' in d.columns else 0

    d['ai_use_case_count'] = d['Q15'].apply(
        lambda v: 0 if pd.isna(v) or 'I have not used AI' in str(v)
        else sum(1 for uc in AI_USE_CASES if uc in str(v))
    ) if 'Q15' in d.columns else 0

    d['ai_concern_count'] = d['Q20'].apply(
        lambda v: 0 if pd.isna(v) or 'No concerns' in str(v)
        else sum(1 for c in CONCERN_ITEMS if c in str(v))
    ) if 'Q20' in d.columns else 0

    d['desired_ai_roles'] = d['Q23'].apply(
        lambda v: 0 if pd.isna(v) or 'should not be used' in str(v)
        else sum(1 for r in AI_ROLES if r in str(v))
    ) if 'Q23' in d.columns else 0

    d['brand_count'] = d['Q5'].apply(
        lambda v: 0 if pd.isna(v) else sum(1 for b in LUXURY_BRANDS if b in str(v))
    ) if 'Q5' in d.columns else 0

    d['is_female'] = (d['Q2'] == 'Female').astype(int)

    d['ai_assistant_freq'] = d['Q18'].map({
        'Never': 0, 'Rarely': 1, 'Occasionally': 2, 'Frequently': 3,
    }).fillna(0)

    d['ai_readiness'] = (
        (d['ai_usage_freq'] / 3) * 0.15 +
        (d['ai_helpfulness'] / 4) * 0.15 +
        (d['ai_desire'] / 4) * 0.20 +
        (d['ai_balance'] / 2) * 0.10 +
        (d['ai_use_case_count'] / 7) * 0.10 +
        (d['ai_for_decision']) * 0.10 +
        (d['desired_ai_roles'] / 5) * 0.10 +
        (d['digital_propensity']) * 0.05 +
        (1 - d['ai_concern_count'] / 4).clip(0, 1) * 0.05
    ).clip(0, 1)

    d['clv'] = d['annual_spend'] * R['avg_luxury_clv_multiplier'] * d['purchase_freq_weight']
    d['retention_improvement'] = R['ai_enhanced_retention_uplift'] * d['ai_readiness'] * d['digital_propensity']
    d['retention_adjusted_clv'] = d['clv'] * (R['baseline_retention_rate'] + d['retention_improvement'])
    d['ai_revenue_uplift'] = d['retention_adjusted_clv'] * R['personalization_revenue_lift'] * d['ai_readiness']
    d['retention_savings'] = d['annual_spend'] * R['churn_cost_multiplier'] * d['retention_improvement']
    d['ai_cost'] = d['annual_spend'] * R['ai_implementation_cost_ratio'] * R['avg_luxury_clv_multiplier']
    d['net_roi'] = (d['ai_revenue_uplift'] + d['retention_savings'] - d['ai_cost']) / d['ai_cost'].replace(0, 1)

    return d


# ── Projection helpers ────────────────────────────────────────────────────

def get_distribution(series, valid_set):
    clean = series.where(series.isin(valid_set)).dropna()
    if len(clean) == 0:
        return {v: 1 / len(valid_set) for v in valid_set}
    return clean.value_counts(normalize=True).to_dict()


def shift_distribution(probs, scale, shift_rate, years_from_baseline):
    n = len(scale)
    p = np.array([probs.get(s, 0.0) for s in scale], dtype=float)
    p = p / p.sum()
    for _ in range(years_from_baseline):
        new_p = p.copy()
        for i in range(n - 1):
            transfer = p[i] * shift_rate
            new_p[i] -= transfer
            new_p[i + 1] += transfer
        p = np.clip(new_p, 0, 1)
        p = p / p.sum()
    return {scale[i]: p[i] for i in range(n)}


def generate_future_cohort(baseline_dists, demo_dists, multi_select_df,
                           year, base_year=2026, shift_rates=None,
                           n_per_year=500, osint=None, seed=42):
    """Generate a synthetic cohort for a given year with shifted AI literacy."""
    rng = np.random.RandomState(seed + year)
    sr = shift_rates or DEFAULT_SHIFT_RATES
    years_from_base = year - base_year
    rows = []

    for _ in range(n_per_year):
        row = {}
        for col, dist in demo_dists.items():
            vals, probs = zip(*dist.items())
            probs = np.array(probs) / sum(probs)
            row[col] = rng.choice(vals, p=probs)

        for col in ORDERED_SCALES:
            shifted = shift_distribution(baseline_dists[col], ORDERED_SCALES[col],
                                          sr.get(col, 0.25), years_from_base)
            vals, probs = zip(*shifted.items())
            probs = np.array(probs)
            probs = probs / probs.sum()
            sampled = rng.choice(vals, p=probs)
            if col == 'Q16':
                row[col] = 'yes' if sampled in ('yes', 'Yes') else 'No'
            else:
                row[col] = sampled

        if multi_select_df is not None and len(multi_select_df) > 0:
            idx = rng.randint(0, len(multi_select_df))
            for col in multi_select_df.columns:
                row[col] = multi_select_df.iloc[idx][col]

        rows.append(row)

    df_year = pd.DataFrame(rows)
    return engineer_features(df_year, osint=osint)
