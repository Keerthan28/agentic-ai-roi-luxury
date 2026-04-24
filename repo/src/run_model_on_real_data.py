"""
Run the trained Agentic AI ROI model on the real survey data.
Dataset: Luxury Goods Shopping Experince_March 11, 2026_17.48.xlsx
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# OSINT: Luxury Industry Customer Retention Benchmarks
# ============================================================================
OSINT_RETENTION = {
    'baseline_retention_rate': 0.82,
    'ai_enhanced_retention_uplift': 0.15,
    'avg_luxury_clv_multiplier': 8.5,
    'churn_cost_multiplier': 5.0,
    'personalization_revenue_lift': 0.20,
    'ai_implementation_cost_ratio': 0.03,
    'digital_engagement_retention_boost': 0.12,
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
    }
}
R = OSINT_RETENTION

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

ai_use_cases = [
    'Product recommendations', 'Comparing prices',
    'Authenticating second-hand items', 'Styling advice',
    'Virtual try-on', 'Customer service chat assistants',
    'Researching brand history or product details',
]
concern_items = ['Lack of personalization', 'Data privacy concerns',
                 'Inaccuracy', 'Removes human touch']
ai_roles = ['Personalization', 'Efficiency', 'Improve authenticity',
            'Assist staff', 'Create immersive digital experiences']
luxury_brands = ['Louis Vuitton', 'Chanel', 'Gucci', 'Hermès', 'Dior',
                 'Prada', 'Rolex', 'Cartier']


def engineer_features(df):
    """Apply full feature engineering pipeline to a dataframe."""
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
        else sum(1 for uc in ai_use_cases if uc in str(v))
    )

    d['ai_concern_count'] = d['Q20'].apply(
        lambda v: 0 if pd.isna(v) or 'No concerns' in str(v)
        else sum(1 for c in concern_items if c in str(v))
    )

    d['desired_ai_roles'] = d['Q23'].apply(
        lambda v: 0 if pd.isna(v) or 'should not be used' in str(v)
        else sum(1 for r in ai_roles if r in str(v))
    )

    d['brand_count'] = d['Q5'].apply(
        lambda v: 0 if pd.isna(v) else sum(1 for b in luxury_brands if b in str(v))
    )

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


# ============================================================================
# 1. LOAD & PREPARE TRAINING DATA (synthetic)
# ============================================================================
print("=" * 70)
print("AGENTIC AI ROI MODEL — REAL SURVEY DATA EVALUATION")
print("=" * 70)

df_train_raw = pd.read_excel('luxury_goods_synthetic_only_ai_positive.xlsx')
for col, valid in VALID_VALUES.items():
    df_train_raw[col] = df_train_raw[col].where(df_train_raw[col].isin(valid))
df_train_raw = df_train_raw[df_train_raw['Q1'].notna()].reset_index(drop=True)
df_train = engineer_features(df_train_raw)
print(f"Training data (synthetic): {len(df_train)} samples")

# ============================================================================
# 2. LOAD & PREPARE REAL SURVEY DATA
# ============================================================================
df_real_raw = pd.read_excel('Luxury Goods Shopping Experince_March 11, 2026_17.48.xlsx')

# Row 0 is the header row with question full text — drop it
df_real_raw = df_real_raw.iloc[1:].reset_index(drop=True)

for col, valid in VALID_VALUES.items():
    if col in df_real_raw.columns:
        df_real_raw[col] = df_real_raw[col].where(df_real_raw[col].isin(valid))

df_real_raw = df_real_raw[df_real_raw['Q1'].notna()].reset_index(drop=True)
df_real = engineer_features(df_real_raw)
print(f"Real survey data: {len(df_real)} samples")

# ============================================================================
# 3. TRAIN MODELS ON SYNTHETIC DATA
# ============================================================================
X_train_full = df_train[FEATURE_COLS]
y_train_full = df_train['net_roi']

models = {
    'XGBoost': XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
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

for name, model in models.items():
    model.fit(X_train_full, y_train_full)

# ============================================================================
# 4. TRAIN/TEST SPLIT ON REAL DATA + CROSS-VALIDATION
# ============================================================================
X_real = df_real[FEATURE_COLS]
y_real = df_real['net_roi']

X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
    X_real, y_real, test_size=0.25, random_state=42
)

print(f"\n--- Real Data Train/Test Split ---")
print(f"Training: {len(X_r_train)}  |  Test: {len(X_r_test)}")

print(f"\n{'=' * 70}")
print("MODEL METRICS — REAL SURVEY DATA")
print(f"{'=' * 70}")

results = {}
for name, model in models.items():
    # Predictions using model trained on synthetic data
    y_pred_full = model.predict(X_real)
    y_pred_test = model.predict(X_r_test)

    # Retrain on real training split for within-real evaluation
    model_retrained = model.__class__(**model.get_params())
    model_retrained.fit(X_r_train, y_r_train)
    y_pred_retrained = model_retrained.predict(X_r_test)

    # Cross-validation on real data (retrained)
    cv_scores = cross_val_score(model_retrained, X_real, y_real, cv=5, scoring='r2')

    # Synthetic-trained model applied to real data
    mae_transfer = mean_absolute_error(y_real, y_pred_full)
    rmse_transfer = np.sqrt(mean_squared_error(y_real, y_pred_full))
    r2_transfer = r2_score(y_real, y_pred_full)

    # Retrained on real train, evaluated on real test
    mae_retrained = mean_absolute_error(y_r_test, y_pred_retrained)
    rmse_retrained = np.sqrt(mean_squared_error(y_r_test, y_pred_retrained))
    r2_retrained = r2_score(y_r_test, y_pred_retrained)

    results[name] = {
        'r2_transfer': r2_transfer, 'mae_transfer': mae_transfer, 'rmse_transfer': rmse_transfer,
        'r2_retrained': r2_retrained, 'mae_retrained': mae_retrained, 'rmse_retrained': rmse_retrained,
        'cv_r2_mean': cv_scores.mean(), 'cv_r2_std': cv_scores.std(),
        'y_pred_full': y_pred_full, 'y_pred_retrained': y_pred_retrained,
        'model_retrained': model_retrained,
    }

    print(f"\n  {name}")
    print(f"  {'─' * 55}")
    print(f"  TRANSFER (synthetic-trained → real data):")
    print(f"    MAE:  {mae_transfer:.4f}   RMSE: {rmse_transfer:.4f}   R²: {r2_transfer:.4f}")
    print(f"  RETRAINED (real train → real test, 75/25 split):")
    print(f"    MAE:  {mae_retrained:.4f}   RMSE: {rmse_retrained:.4f}   R²: {r2_retrained:.4f}")
    print(f"  CROSS-VALIDATION (5-fold on real data):")
    print(f"    R²:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

best_name = max(results, key=lambda k: results[k]['r2_transfer'])
best = results[best_name]
print(f"\n  ► Best transfer model: {best_name} (R² = {best['r2_transfer']:.4f})")

# ============================================================================
# 5. FEATURE IMPORTANCE ON REAL DATA
# ============================================================================
best_retrained = best['model_retrained']
if hasattr(best_retrained, 'feature_importances_'):
    importances = best_retrained.feature_importances_
else:
    importances = np.zeros(len(FEATURE_COLS))

feat_imp = pd.DataFrame({
    'feature': FEATURE_COLS, 'importance': importances,
}).sort_values('importance', ascending=False)

print(f"\n{'=' * 70}")
print(f"FEATURE IMPORTANCE — REAL DATA ({best_name}, retrained)")
print(f"{'=' * 70}")
for _, row in feat_imp.iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")

# ============================================================================
# 6. ROI ANALYSIS — REAL SURVEY RESPONDENTS
# ============================================================================
print(f"\n{'=' * 70}")
print("ROI ANALYSIS — REAL SURVEY DATA (OSINT RETENTION-INFORMED)")
print(f"{'=' * 70}")

print(f"\n--- Real Data ROI Distribution ---")
print(f"  Mean Net ROI:   {df_real['net_roi'].mean():.2f}x")
print(f"  Median Net ROI: {df_real['net_roi'].median():.2f}x")
print(f"  Std Dev:        {df_real['net_roi'].std():.2f}")
print(f"  Min:            {df_real['net_roi'].min():.2f}x  |  Max: {df_real['net_roi'].max():.2f}x")

print(f"\n--- ROI by Income Tier ---")
income_roi = df_real.groupby('Q3').agg(
    mean_roi=('net_roi', 'mean'), mean_clv=('clv', 'mean'),
    mean_retention_improvement=('retention_improvement', 'mean'),
    mean_ai_readiness=('ai_readiness', 'mean'), count=('net_roi', 'count'),
).sort_values('mean_roi', ascending=False)
print(income_roi.to_string())

print(f"\n--- ROI by Age Group ---")
age_roi = df_real.groupby('Q1').agg(
    mean_roi=('net_roi', 'mean'),
    mean_ai_readiness=('ai_readiness', 'mean'),
    mean_retention_improvement=('retention_improvement', 'mean'),
    count=('net_roi', 'count'),
).reindex(['Under 18', '18–24', '25–34', '35–44', '45–54', '55–64', '65+']).dropna()
print(age_roi.to_string())

print(f"\n--- AI Adoption Profile ---")
ai_bins = pd.cut(df_real['ai_readiness'], bins=[0, 0.25, 0.5, 0.75, 1.0],
                  labels=['Low', 'Medium', 'High', 'Very High'])
print(df_real.groupby(ai_bins, observed=True).agg(
    mean_roi=('net_roi', 'mean'), mean_revenue_uplift=('ai_revenue_uplift', 'mean'),
    mean_retention_savings=('retention_savings', 'mean'), count=('net_roi', 'count'),
).to_string())

print(f"\n--- Retention Impact (OSINT-Informed) ---")
print(f"  Baseline retention:              {R['baseline_retention_rate']:.0%}")
print(f"  Avg retention improvement:       +{df_real['retention_improvement'].mean():.1%}")
print(f"  Projected post-AI retention:     {R['baseline_retention_rate'] + df_real['retention_improvement'].mean():.1%}")
print(f"  Avg revenue uplift / customer:   ${df_real['ai_revenue_uplift'].mean():,.0f}")
print(f"  Avg retention savings / customer:${df_real['retention_savings'].mean():,.0f}")
print(f"  Avg AI cost / customer:          ${df_real['ai_cost'].mean():,.0f}")

n = 1000
total_rev = df_real['ai_revenue_uplift'].mean() * n
total_ret = df_real['retention_savings'].mean() * n
total_cost = df_real['ai_cost'].mean() * n
total_net = total_rev + total_ret - total_cost
print(f"\n--- Projected Impact (per {n:,} customers) ---")
print(f"  Revenue uplift:       ${total_rev:>12,.0f}")
print(f"  Retention savings:    ${total_ret:>12,.0f}")
print(f"  AI cost:              ${total_cost:>12,.0f}")
print(f"  ──────────────────────────────────")
print(f"  NET BENEFIT:          ${total_net:>12,.0f}")
print(f"  ROI:                   {total_net / total_cost:.1f}x")

# ============================================================================
# 7. COMPARATIVE METRICS TABLE
# ============================================================================
print(f"\n{'=' * 70}")
print("SYNTHETIC vs REAL — SIDE-BY-SIDE COMPARISON")
print(f"{'=' * 70}")
print(f"  {'Metric':<35s} {'Synthetic':>12s} {'Real':>12s}")
print(f"  {'─' * 60}")
print(f"  {'Sample size':<35s} {len(df_train):>12d} {len(df_real):>12d}")
print(f"  {'Mean Net ROI':<35s} {df_train['net_roi'].mean():>12.2f}x {df_real['net_roi'].mean():>12.2f}x")
print(f"  {'Median Net ROI':<35s} {df_train['net_roi'].median():>12.2f}x {df_real['net_roi'].median():>12.2f}x")
print(f"  {'Mean AI Readiness':<35s} {df_train['ai_readiness'].mean():>12.3f} {df_real['ai_readiness'].mean():>12.3f}")
print(f"  {'Mean Retention Improvement':<35s} {df_train['retention_improvement'].mean():>11.1%} {df_real['retention_improvement'].mean():>11.1%}")
print(f"  {'Mean CLV ($)':<35s} {df_train['clv'].mean():>12,.0f} {df_real['clv'].mean():>12,.0f}")
print(f"  {'Avg Revenue Uplift ($/cust)':<35s} {df_train['ai_revenue_uplift'].mean():>12,.0f} {df_real['ai_revenue_uplift'].mean():>12,.0f}")
print(f"  {'Avg Retention Savings ($/cust)':<35s} {df_train['retention_savings'].mean():>12,.0f} {df_real['retention_savings'].mean():>12,.0f}")
print(f"  {'AI-ready (>0.5) share':<35s} {(df_train['ai_readiness']>0.5).mean():>12.0%} {(df_real['ai_readiness']>0.5).mean():>12.0%}")

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================
fig, axes = plt.subplots(3, 3, figsize=(20, 18))
fig.suptitle('Agentic AI ROI — Real Survey Data Analysis\n(Model Trained on Synthetic, Applied to Real)',
             fontsize=16, fontweight='bold', y=1.01)
plt.subplots_adjust(hspace=0.35, wspace=0.30)

# 8a. ROI distribution comparison
ax = axes[0, 0]
ax.hist(df_train['net_roi'], bins=20, alpha=0.5, color='#3498db', label=f"Synthetic (n={len(df_train)})", density=True)
ax.hist(df_real['net_roi'], bins=12, alpha=0.7, color='#e74c3c', label=f"Real (n={len(df_real)})", density=True)
ax.axvline(df_real['net_roi'].mean(), color='#e74c3c', linestyle='--', lw=2)
ax.axvline(df_train['net_roi'].mean(), color='#3498db', linestyle='--', lw=2)
ax.set_xlabel('Net ROI (x)')
ax.set_ylabel('Density')
ax.set_title('ROI Distribution: Synthetic vs Real')
ax.legend()

# 8b. Transfer: Actual vs Predicted on real data
ax = axes[0, 1]
ax.scatter(y_real, best['y_pred_full'], alpha=0.7, color='#2c3e50', edgecolors='white', s=70)
mn, mx = min(y_real.min(), best['y_pred_full'].min()), max(y_real.max(), best['y_pred_full'].max())
ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect')
ax.set_xlabel('Actual ROI')
ax.set_ylabel('Predicted ROI (transfer)')
ax.set_title(f'Transfer: Actual vs Predicted (R²={best["r2_transfer"]:.3f})')
ax.legend()

# 8c. Feature importance (real, retrained)
ax = axes[0, 2]
top = feat_imp.head(12)
colors_fi = plt.cm.RdYlGn(np.linspace(0.3, 0.95, len(top)))[::-1]
ax.barh(top['feature'], top['importance'], color=colors_fi, edgecolor='white')
ax.set_xlabel('Importance')
ax.set_title(f'Feature Importance — Real Data ({best_name})')
ax.invert_yaxis()

# 8d. ROI vs AI Readiness (real)
ax = axes[1, 0]
sc = ax.scatter(df_real['ai_readiness'], df_real['net_roi'],
                c=df_real['annual_spend'], cmap='YlOrRd', alpha=0.8, edgecolors='white', s=80)
plt.colorbar(sc, ax=ax, label='Annual Spend ($)')
ax.set_xlabel('AI Readiness Score')
ax.set_ylabel('Net ROI (x)')
ax.set_title('ROI vs AI Readiness — Real Respondents')

# 8e. Retention uplift by age (real)
ax = axes[1, 1]
seg = df_real.groupby('Q1')['retention_improvement'].mean().reindex(
    ['Under 18', '18–24', '25–34', '35–44', '45–54', '55–64']
).dropna()
c_seg = plt.cm.Blues(np.linspace(0.4, 0.9, len(seg)))
bars = ax.bar(seg.index, seg.values * 100, color=c_seg, edgecolor='white', lw=1.5)
ax.set_xlabel('Age Group')
ax.set_ylabel('Retention Improvement (%)')
ax.set_title('AI-Driven Retention Uplift by Age — Real')
ax.tick_params(axis='x', rotation=30)
for b, v in zip(bars, seg.values):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
            f'{v*100:.1f}%', ha='center', va='bottom', fontsize=9)

# 8f. Model comparison (transfer R²)
ax = axes[1, 2]
names = list(results.keys())
r2_t = [results[m]['r2_transfer'] for m in names]
r2_r = [results[m]['r2_retrained'] for m in names]
cv_r = [results[m]['cv_r2_mean'] for m in names]
x = np.arange(len(names))
w = 0.25
ax.bar(x - w, r2_t, w, label='Transfer R²', color='#3498db', edgecolor='white')
ax.bar(x, r2_r, w, label='Retrained R²', color='#27ae60', edgecolor='white')
ax.bar(x + w, cv_r, w, label='CV R² (real)', color='#e74c3c', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15)
ax.set_ylabel('R² Score')
ax.set_title('Model Comparison on Real Data')
ax.legend(fontsize=8)
ax.set_ylim(0, 1.1)
for i, (t, r, c) in enumerate(zip(r2_t, r2_r, cv_r)):
    ax.text(i - w, t + 0.02, f'{t:.3f}', ha='center', fontsize=7)
    ax.text(i, r + 0.02, f'{r:.3f}', ha='center', fontsize=7)
    ax.text(i + w, c + 0.02, f'{c:.3f}', ha='center', fontsize=7)

# 8g. ROI breakdown waterfall (real)
ax = axes[2, 0]
cats = ['Revenue\nUplift', 'Retention\nSavings', 'AI Cost\n(negative)', 'Net\nBenefit']
vals = [df_real['ai_revenue_uplift'].mean(), df_real['retention_savings'].mean(),
        -df_real['ai_cost'].mean(),
        df_real['ai_revenue_uplift'].mean() + df_real['retention_savings'].mean() - df_real['ai_cost'].mean()]
col_wf = ['#27ae60', '#2ecc71', '#e74c3c', '#3498db']
bars = ax.bar(cats, vals, color=col_wf, edgecolor='white', lw=1.5)
ax.axhline(y=0, color='black', lw=0.5)
ax.set_ylabel('$ per Customer')
ax.set_title('ROI Breakdown — Real Respondents')
for b, v in zip(bars, vals):
    yp = b.get_height() + (30 if v > 0 else -80)
    ax.text(b.get_x() + b.get_width()/2, yp, f'${v:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 8h. AI Readiness distribution comparison
ax = axes[2, 1]
ax.hist(df_train['ai_readiness'], bins=20, alpha=0.5, color='#3498db', label='Synthetic', density=True)
ax.hist(df_real['ai_readiness'], bins=10, alpha=0.7, color='#e74c3c', label='Real', density=True)
ax.set_xlabel('AI Readiness Score')
ax.set_ylabel('Density')
ax.set_title('AI Readiness: Synthetic vs Real')
ax.legend()

# 8i. Correlation heatmap (real)
ax = axes[2, 2]
corr_cols = ['ai_readiness', 'satisfaction_score', 'ai_helpfulness',
             'ai_desire', 'retention_improvement', 'net_roi', 'clv']
corr_matrix = df_real[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlations — Real Data')
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('real_data_roi_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"\n► Visualizations saved to: real_data_roi_analysis.png")

# ============================================================================
# 9. EXECUTIVE SUMMARY
# ============================================================================
print(f"\n{'=' * 70}")
print("EXECUTIVE SUMMARY — REAL SURVEY DATA")
print(f"{'=' * 70}")
print(f"""
REAL DATA MODEL METRICS
───────────────────────
Respondents: {len(df_real)} (real survey, post-cleaning)
Train/Test:  {len(X_r_train)} / {len(X_r_test)} (75/25 split)

TRANSFER LEARNING (synthetic-trained → real):
  Best: {best_name}  |  R² = {best['r2_transfer']:.3f}  |  MAE = {best['mae_transfer']:.3f}

RETRAINED ON REAL DATA (75/25 split):
  Best: {best_name}  |  R² = {best['r2_retrained']:.3f}  |  MAE = {best['mae_retrained']:.3f}

5-FOLD CROSS-VALIDATION (real data):
  R² = {best['cv_r2_mean']:.3f} ± {best['cv_r2_std']:.3f}

ROI PROJECTIONS (real respondents):
  • Mean Net ROI:                {df_real['net_roi'].mean():.2f}x
  • Retention improvement:       +{df_real['retention_improvement'].mean():.1%} (82% → {82 + df_real['retention_improvement'].mean()*100:.1f}%)
  • Revenue uplift / customer:   ${df_real['ai_revenue_uplift'].mean():,.0f}
  • Retention savings / customer: ${df_real['retention_savings'].mean():,.0f}
  • Net benefit / 1000 customers: ${total_net:,.0f}

TOP ROI DRIVERS: {', '.join(feat_imp.head(3)['feature'].tolist())}
""")
print("=" * 70)
