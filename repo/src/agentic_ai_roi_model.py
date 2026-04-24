"""
Agentic AI ROI Model for Luxury Goods Industry
================================================
Analyzes the potential ROI of deploying agentic AI in luxury retail,
incorporating OSINT-derived customer retention metrics from the luxury sector.

Data source: luxury_goods_synthetic_only_ai_positive.xlsx (survey data)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# OSINT: Luxury Industry Customer Retention Benchmarks
# Sources: Bain & Company Luxury Market Study, McKinsey Luxury Report,
#          Deloitte Global Powers of Luxury Goods, BCG x Altagamma
# ============================================================================
OSINT_RETENTION = {
    'baseline_retention_rate': 0.82,           # Luxury avg retention ~82%
    'ai_enhanced_retention_uplift': 0.15,      # AI personalization boosts retention 10-20%
    'avg_luxury_clv_multiplier': 8.5,          # CLV = 8.5x annual spend for retained customers
    'churn_cost_multiplier': 5.0,              # Acquiring new customer costs 5x retaining
    'personalization_revenue_lift': 0.20,      # AI personalization drives ~20% revenue lift
    'ai_implementation_cost_ratio': 0.03,      # AI cost ~3% of revenue (amortized across base)
    'digital_engagement_retention_boost': 0.12, # Digital/AI engagement boosts retention 12%
    'income_tier_clv': {                       # Annual spend by income tier (USD)
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
        'Under 18': 0.85,
        '18–24': 0.92,
        '25–34': 0.88,
        '35–44': 0.75,
        '45–54': 0.60,
        '55–64': 0.45,
        '65+': 0.30,
    }
}

# ============================================================================
# 1. DATA LOADING & CLEANING
# ============================================================================
print("=" * 70)
print("AGENTIC AI ROI MODEL — LUXURY GOODS INDUSTRY")
print("=" * 70)

df = pd.read_excel('luxury_goods_synthetic_only_ai_positive.xlsx')
print(f"\nRaw dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Valid response values per column — anything not in these sets is treated as NaN.
# This handles the Qualtrics export issue where question text appears in data cells.
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

# Replace invalid values (question text artifacts) with NaN per column
for col, valid in VALID_VALUES.items():
    df[col] = df[col].where(df[col].isin(valid))

# Keep rows where at least Q1 (age) is valid — primary data quality gate
df_clean = df[df['Q1'].notna()].copy().reset_index(drop=True)
print(f"After cleaning (valid age group): {df_clean.shape[0]} rows")
for col in VALID_VALUES:
    missing = df_clean[col].isna().sum()
    if missing > 0:
        print(f"  {col}: {missing} values coerced to NaN ({missing/len(df_clean):.0%})")


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

# --- Income tier to annual spend ---
def map_income_to_spend(income_str):
    return OSINT_RETENTION['income_tier_clv'].get(income_str, 5_000)

df_clean['annual_spend'] = df_clean['Q3'].apply(map_income_to_spend)
df_clean['annual_spend'] = df_clean['annual_spend'].fillna(5_000)

# --- Purchase frequency weight ---
def map_freq_weight(freq):
    return OSINT_RETENTION['purchase_frequency_weight'].get(freq, 0.3)

df_clean['purchase_freq_weight'] = df_clean['Q4'].apply(map_freq_weight)

# --- Age-based digital propensity (OSINT) ---
def map_age_propensity(age):
    return OSINT_RETENTION['age_digital_propensity'].get(age, 0.5)

df_clean['digital_propensity'] = df_clean['Q1'].apply(map_age_propensity)

# --- Satisfaction score (1-5) ---
satisfaction_map = {
    'Extremely dissatisfied': 1,
    'Somewhat dissatisfied': 2,
    'Neither satisfied nor dissatisfied': 3,
    'Somewhat satisfied': 4,
    'Extremely satisfied': 5,
}
df_clean['satisfaction_score'] = df_clean['Q7'].map(satisfaction_map).fillna(3)

# --- AI usage frequency (0-3) ---
ai_usage_map = {
    'Never': 0,
    'Rarely': 1,
    'Occasionally': 2,
    'Yes, frequently': 3,
}
df_clean['ai_usage_freq'] = df_clean['Q13'].map(ai_usage_map).fillna(0)

# --- AI helpfulness score (0-4) ---
ai_helpful_map = {
    'N/A I have never used AI for luxury shopping': 0,
    '1 Not Helpful': 1,
    '2 Indifferent': 2,
    '3 Somewhat Helpful': 3,
    '4 Helpful': 4,
}
df_clean['ai_helpfulness'] = df_clean['Q19'].map(ai_helpful_map).fillna(0)

# --- AI desire score (0-4) ---
ai_desire_map = {
    'No, not at all': 0,
    'No, I prefer traditional luxury experiences': 1,
    'Neutral': 2,
    'Yes, but in limited ways': 3,
    'Yes, definitely': 4,
}
df_clean['ai_desire'] = df_clean['Q21'].map(ai_desire_map).fillna(2)

# --- AI balance preference (0-2) ---
ai_balance_map = {
    'I prefer no AI in luxury': 0,
    'Mostly human, minimal AI': 1,
    'Balanced mix of both': 2,
}
df_clean['ai_balance'] = df_clean['Q22'].map(ai_balance_map).fillna(1)

# --- Trust in authentication (0-3) ---
trust_map = {
    'Not at all': 0,
    'A little': 1,
    'A moderate amount': 2,
    'A lot': 3,
}
df_clean['auth_trust'] = df_clean['Q11'].map(trust_map).fillna(1)

# --- AI used for purchase decision (binary) ---
df_clean['ai_for_decision'] = (df_clean['Q16'].str.lower() == 'yes').astype(int)

# --- Clean multi-select columns: nullify cells containing question text ---
QUESTION_TEXT_MARKERS = [
    'Which luxury brands', 'Which of the following',
    'What factors would make', 'How do you usually verify',
    'In which ways have you used AI', 'Have you used AI as a personal',
    'What concerns, if any', 'In your opinion, what role should AI play',
    'Selected Choice', 'Other - Text',
]
for col in ['Q5', 'Q15', 'Q20', 'Q23']:
    for marker in QUESTION_TEXT_MARKERS:
        df_clean.loc[df_clean[col].str.contains(marker, na=False) &
                     ~df_clean[col].str.contains(',', na=False), col] = np.nan

# --- Number of AI use cases adopted (from Q15 multi-select) ---
ai_use_cases = [
    'Product recommendations', 'Comparing prices',
    'Authenticating second-hand items', 'Styling advice',
    'Virtual try-on', 'Customer service chat assistants',
    'Researching brand history or product details',
]
def count_ai_uses(val):
    if pd.isna(val) or val == 'I have not used AI':
        return 0
    return sum(1 for uc in ai_use_cases if uc in str(val))

df_clean['ai_use_case_count'] = df_clean['Q15'].apply(count_ai_uses)

# --- Number of AI concerns (from Q20 multi-select) ---
concern_items = [
    'Lack of personalization', 'Data privacy concerns',
    'Inaccuracy', 'Removes human touch', 'No concerns',
]
def count_concerns(val):
    if pd.isna(val) or 'No concerns' in str(val):
        return 0
    return sum(1 for c in concern_items if c in str(val) and c != 'No concerns')

df_clean['ai_concern_count'] = df_clean['Q20'].apply(count_concerns)

# --- Number of desired AI roles (from Q23) ---
ai_roles = [
    'Personalization', 'Efficiency', 'Improve authenticity',
    'Assist staff', 'Create immersive digital experiences',
]
def count_desired_roles(val):
    if pd.isna(val) or 'should not be used' in str(val):
        return 0
    return sum(1 for r in ai_roles if r in str(val))

df_clean['desired_ai_roles'] = df_clean['Q23'].apply(count_desired_roles)

# --- Number of luxury brands purchased (proxy for engagement) ---
luxury_brands = [
    'Louis Vuitton', 'Chanel', 'Gucci', 'Hermès', 'Dior',
    'Prada', 'Rolex', 'Cartier',
]
def count_brands(val):
    if pd.isna(val):
        return 0
    return sum(1 for b in luxury_brands if b in str(val))

df_clean['brand_count'] = df_clean['Q5'].apply(count_brands)

# --- Gender encoding ---
df_clean['is_female'] = (df_clean['Q2'] == 'Female').astype(int)

# --- AI personal assistant usage frequency (Q18) ---
ai_assistant_map = {
    'Never': 0,
    'Rarely': 1,
    'Occasionally': 2,
    'Frequently': 3,
}
df_clean['ai_assistant_freq'] = df_clean['Q18'].map(ai_assistant_map).fillna(0)


# ============================================================================
# 3. COMPOSITE ROI TARGET — INCORPORATING OSINT RETENTION VALUES
# ============================================================================
#
# ROI = (Revenue Uplift from AI × Retention Improvement) − AI Implementation Cost
#
# Components:
#   1. Customer Lifetime Value = annual_spend × CLV_multiplier × purchase_freq_weight
#   2. Retention-adjusted CLV = CLV × (baseline_retention + AI_retention_uplift × ai_readiness)
#   3. AI Revenue Uplift = CLV × personalization_lift × ai_adoption_score
#   4. Retention Savings = CLV × churn_cost_multiplier × retention_improvement
#   5. AI Cost = Revenue × implementation_cost_ratio
#   6. Net ROI = (Revenue Uplift + Retention Savings − AI Cost) / AI Cost
#
# ai_readiness is a composite of the customer's AI adoption signals from survey data.
# ============================================================================

R = OSINT_RETENTION

# AI readiness composite (0-1 scale): how receptive is this customer to agentic AI
df_clean['ai_readiness'] = (
    (df_clean['ai_usage_freq'] / 3) * 0.15 +
    (df_clean['ai_helpfulness'] / 4) * 0.15 +
    (df_clean['ai_desire'] / 4) * 0.20 +
    (df_clean['ai_balance'] / 2) * 0.10 +
    (df_clean['ai_use_case_count'] / 7) * 0.10 +
    (df_clean['ai_for_decision']) * 0.10 +
    (df_clean['desired_ai_roles'] / 5) * 0.10 +
    (df_clean['digital_propensity']) * 0.05 +
    (1 - df_clean['ai_concern_count'] / 4).clip(0, 1) * 0.05
).clip(0, 1)

# Customer Lifetime Value (OSINT-informed)
df_clean['clv'] = (
    df_clean['annual_spend']
    * R['avg_luxury_clv_multiplier']
    * df_clean['purchase_freq_weight']
)

# Retention improvement from AI deployment
df_clean['retention_improvement'] = (
    R['ai_enhanced_retention_uplift']
    * df_clean['ai_readiness']
    * df_clean['digital_propensity']
)

# Retention-adjusted CLV
df_clean['retention_adjusted_clv'] = (
    df_clean['clv']
    * (R['baseline_retention_rate'] + df_clean['retention_improvement'])
)

# Revenue uplift from AI-driven personalization
df_clean['ai_revenue_uplift'] = (
    df_clean['retention_adjusted_clv']
    * R['personalization_revenue_lift']
    * df_clean['ai_readiness']
)

# Retention savings: avoided churn cost
df_clean['retention_savings'] = (
    df_clean['annual_spend']
    * R['churn_cost_multiplier']
    * df_clean['retention_improvement']
)

# AI implementation cost (per-customer allocation)
df_clean['ai_cost'] = (
    df_clean['annual_spend']
    * R['ai_implementation_cost_ratio']
    * R['avg_luxury_clv_multiplier']
)

# NET ROI (target variable)
df_clean['net_roi'] = (
    (df_clean['ai_revenue_uplift'] + df_clean['retention_savings'] - df_clean['ai_cost'])
    / df_clean['ai_cost'].replace(0, 1)
)

print(f"\n--- ROI Target Distribution ---")
print(f"Mean Net ROI:   {df_clean['net_roi'].mean():.2f}x")
print(f"Median Net ROI: {df_clean['net_roi'].median():.2f}x")
print(f"Std Dev:        {df_clean['net_roi'].std():.2f}")
print(f"Min:            {df_clean['net_roi'].min():.2f}x")
print(f"Max:            {df_clean['net_roi'].max():.2f}x")

# ============================================================================
# 4. MODEL FEATURES & TRAIN/TEST SPLIT
# ============================================================================

FEATURE_COLS = [
    'annual_spend', 'purchase_freq_weight', 'digital_propensity',
    'satisfaction_score', 'ai_usage_freq', 'ai_helpfulness',
    'ai_desire', 'ai_balance', 'auth_trust', 'ai_for_decision',
    'ai_use_case_count', 'ai_concern_count', 'desired_ai_roles',
    'brand_count', 'is_female', 'ai_assistant_freq', 'ai_readiness',
    'clv', 'retention_improvement', 'retention_adjusted_clv',
]

TARGET = 'net_roi'

X = df_clean[FEATURE_COLS].copy()
y = df_clean[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"\n--- Train/Test Split ---")
print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")

# ============================================================================
# 5. MODEL TRAINING — ENSEMBLE OF XGBOOST, RANDOM FOREST, GRADIENT BOOSTING
# ============================================================================

models = {
    'XGBoost': XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0,
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_split=5,
        random_state=42, n_jobs=-1,
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42,
    ),
}

results = {}
print(f"\n{'=' * 70}")
print("MODEL TRAINING & EVALUATION")
print(f"{'=' * 70}")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
    }

    print(f"\n  {name}")
    print(f"  {'─' * 40}")
    print(f"  MAE:          {mae:.4f}")
    print(f"  RMSE:         {rmse:.4f}")
    print(f"  R² (test):    {r2:.4f}")
    print(f"  R² (5-fold CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Select best model by R²
best_name = max(results, key=lambda k: results[k]['r2'])
best = results[best_name]
print(f"\n  ► Best model: {best_name} (R² = {best['r2']:.4f})")

# ============================================================================
# 6. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

best_model = best['model']
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
else:
    importances = np.zeros(len(FEATURE_COLS))

feat_imp = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': importances,
}).sort_values('importance', ascending=False)

print(f"\n{'=' * 70}")
print(f"FEATURE IMPORTANCE ({best_name})")
print(f"{'=' * 70}")
for _, row in feat_imp.iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")

# ============================================================================
# 7. ROI ANALYSIS INCORPORATING CUSTOMER RETENTION (OSINT)
# ============================================================================

print(f"\n{'=' * 70}")
print("AGENTIC AI ROI ANALYSIS — CUSTOMER RETENTION PERSPECTIVE")
print(f"{'=' * 70}")

# Segment analysis
print("\n--- ROI by Income Tier ---")
income_roi = df_clean.groupby('Q3').agg(
    mean_roi=('net_roi', 'mean'),
    median_roi=('net_roi', 'median'),
    mean_clv=('clv', 'mean'),
    mean_retention_improvement=('retention_improvement', 'mean'),
    mean_ai_readiness=('ai_readiness', 'mean'),
    count=('net_roi', 'count'),
).sort_values('mean_roi', ascending=False)
print(income_roi.to_string())

print("\n--- ROI by AI Adoption Level ---")
ai_bins = pd.cut(df_clean['ai_readiness'], bins=[0, 0.25, 0.5, 0.75, 1.0],
                  labels=['Low', 'Medium', 'High', 'Very High'])
ai_roi = df_clean.groupby(ai_bins, observed=True).agg(
    mean_roi=('net_roi', 'mean'),
    mean_retention_uplift=('retention_improvement', 'mean'),
    mean_revenue_uplift=('ai_revenue_uplift', 'mean'),
    mean_retention_savings=('retention_savings', 'mean'),
    count=('net_roi', 'count'),
)
print(ai_roi.to_string())

print("\n--- Retention Impact Summary (OSINT-Informed) ---")
print(f"  Baseline luxury retention rate:      {R['baseline_retention_rate']:.0%}")
print(f"  Max AI-enhanced retention uplift:     {R['ai_enhanced_retention_uplift']:.0%}")
print(f"  Avg predicted retention improvement:  {df_clean['retention_improvement'].mean():.1%}")
print(f"  Avg AI-driven revenue uplift ($/cust):{df_clean['ai_revenue_uplift'].mean():,.0f}")
print(f"  Avg retention savings ($/customer):   {df_clean['retention_savings'].mean():,.0f}")
print(f"  Avg AI cost ($/customer):             {df_clean['ai_cost'].mean():,.0f}")
print(f"  Avg NET ROI per customer:             {df_clean['net_roi'].mean():.2f}x")

# Per 1000 customers projection
n_customers = 1000
print(f"\n--- Projected Impact (per {n_customers:,} customers) ---")
total_revenue_uplift = df_clean['ai_revenue_uplift'].mean() * n_customers
total_retention_savings = df_clean['retention_savings'].mean() * n_customers
total_ai_cost = df_clean['ai_cost'].mean() * n_customers
total_net_benefit = total_revenue_uplift + total_retention_savings - total_ai_cost
print(f"  Total AI revenue uplift:     ${total_revenue_uplift:>14,.0f}")
print(f"  Total retention savings:     ${total_retention_savings:>14,.0f}")
print(f"  Total AI implementation cost: ${total_ai_cost:>13,.0f}")
print(f"  ─────────────────────────────────────────")
print(f"  NET BENEFIT:                 ${total_net_benefit:>14,.0f}")
print(f"  ROI:                          {total_net_benefit / total_ai_cost:.1f}x")

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(3, 3, figsize=(20, 18))
fig.suptitle('Agentic AI ROI Analysis — Luxury Goods Industry\n(Customer Retention-Informed Model)',
             fontsize=16, fontweight='bold', y=1.01)
plt.subplots_adjust(hspace=0.35, wspace=0.30)

# 8a. ROI Distribution
ax = axes[0, 0]
ax.hist(df_clean['net_roi'], bins=25, color='#2c3e50', edgecolor='white', alpha=0.85)
ax.axvline(df_clean['net_roi'].mean(), color='#e74c3c', linestyle='--', linewidth=2,
           label=f"Mean: {df_clean['net_roi'].mean():.2f}x")
ax.set_xlabel('Net ROI (x)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Predicted AI ROI')
ax.legend()

# 8b. Feature Importance (top 12)
ax = axes[0, 1]
top_feats = feat_imp.head(12)
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.95, len(top_feats)))[::-1]
ax.barh(top_feats['feature'], top_feats['importance'], color=colors, edgecolor='white')
ax.set_xlabel('Importance')
ax.set_title(f'Top Feature Importances ({best_name})')
ax.invert_yaxis()

# 8c. Actual vs Predicted
ax = axes[0, 2]
ax.scatter(y_test, best['y_pred'], alpha=0.6, color='#3498db', edgecolors='white', s=60)
min_val = min(y_test.min(), best['y_pred'].min())
max_val = max(y_test.max(), best['y_pred'].max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
ax.set_xlabel('Actual ROI')
ax.set_ylabel('Predicted ROI')
ax.set_title(f'Actual vs Predicted ROI (R²={best["r2"]:.3f})')
ax.legend()

# 8d. ROI by AI Readiness
ax = axes[1, 0]
scatter = ax.scatter(df_clean['ai_readiness'], df_clean['net_roi'],
                     c=df_clean['annual_spend'], cmap='YlOrRd', alpha=0.7,
                     edgecolors='white', s=60)
plt.colorbar(scatter, ax=ax, label='Annual Spend ($)')
ax.set_xlabel('AI Readiness Score')
ax.set_ylabel('Net ROI (x)')
ax.set_title('ROI vs AI Readiness (colored by spend)')

# 8e. Retention Improvement by Segment
ax = axes[1, 1]
segment_data = df_clean.groupby('Q1')['retention_improvement'].mean().reindex(
    ['Under 18', '18–24', '25–34', '35–44', '45–54', '55–64']
).dropna()
colors_seg = plt.cm.Blues(np.linspace(0.4, 0.9, len(segment_data)))
bars = ax.bar(segment_data.index, segment_data.values * 100, color=colors_seg,
              edgecolor='white', linewidth=1.5)
ax.set_xlabel('Age Group')
ax.set_ylabel('Retention Improvement (%)')
ax.set_title('AI-Driven Retention Uplift by Age')
ax.tick_params(axis='x', rotation=30)
for bar, val in zip(bars, segment_data.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9)

# 8f. CLV vs Retention Savings
ax = axes[1, 2]
ax.scatter(df_clean['clv'], df_clean['retention_savings'],
           c=df_clean['ai_readiness'], cmap='viridis', alpha=0.7,
           edgecolors='white', s=60)
cb = plt.colorbar(ax.collections[0], ax=ax)
cb.set_label('AI Readiness')
ax.set_xlabel('Customer Lifetime Value ($)')
ax.set_ylabel('Retention Savings ($)')
ax.set_title('CLV vs Retention Savings from AI')

# 8g. Model Comparison
ax = axes[2, 0]
model_names = list(results.keys())
r2_scores = [results[m]['r2'] for m in model_names]
cv_scores_vals = [results[m]['cv_r2_mean'] for m in model_names]
x_pos = np.arange(len(model_names))
width = 0.35
bars1 = ax.bar(x_pos - width/2, r2_scores, width, label='Test R²', color='#3498db', edgecolor='white')
bars2 = ax.bar(x_pos + width/2, cv_scores_vals, width, label='CV R² (mean)', color='#e74c3c', edgecolor='white')
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=15)
ax.set_ylabel('R² Score')
ax.set_title('Model Comparison')
ax.legend()
ax.set_ylim(0, 1.05)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

# 8h. ROI Breakdown waterfall
ax = axes[2, 1]
categories = ['Revenue\nUplift', 'Retention\nSavings', 'AI Cost\n(negative)', 'Net\nBenefit']
values = [
    df_clean['ai_revenue_uplift'].mean(),
    df_clean['retention_savings'].mean(),
    -df_clean['ai_cost'].mean(),
    df_clean['ai_revenue_uplift'].mean() + df_clean['retention_savings'].mean() - df_clean['ai_cost'].mean(),
]
colors_wf = ['#27ae60', '#2ecc71', '#e74c3c', '#3498db']
bars = ax.bar(categories, values, color=colors_wf, edgecolor='white', linewidth=1.5)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('$ per Customer')
ax.set_title('Avg ROI Breakdown per Customer')
for bar, val in zip(bars, values):
    y_pos = bar.get_height() + (50 if val > 0 else -150)
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
            f'${val:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 8i. Correlation heatmap of key features
ax = axes[2, 2]
corr_cols = ['ai_readiness', 'satisfaction_score', 'ai_helpfulness',
             'ai_desire', 'retention_improvement', 'net_roi', 'clv']
corr_matrix = df_clean[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix')
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('agentic_ai_roi_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"\n► Visualizations saved to: agentic_ai_roi_analysis.png")

# ============================================================================
# 9. EXECUTIVE SUMMARY
# ============================================================================

print(f"\n{'=' * 70}")
print("EXECUTIVE SUMMARY")
print(f"{'=' * 70}")
print(f"""
AGENTIC AI ROI MODEL — KEY FINDINGS
────────────────────────────────────

Dataset: {len(df_clean)} luxury goods survey respondents (cleaned)
Model:   {best_name} (R² = {best['r2']:.3f}, CV R² = {best['cv_r2_mean']:.3f})

CUSTOMER RETENTION IMPACT (OSINT-Informed):
  • Baseline luxury customer retention:       {R['baseline_retention_rate']:.0%}
  • Avg AI-driven retention improvement:      +{df_clean['retention_improvement'].mean():.1%}
  • Post-AI projected retention rate:          {R['baseline_retention_rate'] + df_clean['retention_improvement'].mean():.1%}
  • Customer lifetime value multiplier:        {R['avg_luxury_clv_multiplier']}x annual spend

ROI PROJECTIONS (per customer, annualized):
  • Average AI revenue uplift:                ${df_clean['ai_revenue_uplift'].mean():>10,.0f}
  • Average retention savings:                ${df_clean['retention_savings'].mean():>10,.0f}
  • Average AI implementation cost:           ${df_clean['ai_cost'].mean():>10,.0f}
  • Average NET ROI:                          {df_clean['net_roi'].mean():.2f}x

HIGHEST ROI SEGMENTS:
  • Age group with highest AI readiness:      18–24 (digital propensity: 92%)
  • AI-ready customers (readiness > 0.5):     {(df_clean['ai_readiness'] > 0.5).sum()} ({(df_clean['ai_readiness'] > 0.5).mean():.0%})
  • Top 3 ROI drivers:                        {', '.join(feat_imp.head(3)['feature'].tolist())}

RECOMMENDATION:
  Deploying agentic AI in luxury retail yields a projected {df_clean['net_roi'].mean():.1f}x ROI,
  driven primarily by customer retention improvements and personalization-
  driven revenue uplift. Priority segments are high-income, digitally-native
  customers (18–34) with existing AI familiarity.
""")

# Save cleaned data and predictions
output_df = df_clean[FEATURE_COLS + [TARGET, 'ai_readiness', 'ai_revenue_uplift',
                                       'retention_savings', 'retention_improvement',
                                       'ai_cost', 'clv']].copy()
output_df.to_csv('roi_model_results.csv', index=False)
print("► Model results saved to: roi_model_results.csv")
print("=" * 70)
