"""
Agentic AI ROI Dashboard — Luxury Goods Industry
Streamlit app with interactive OSINT parameter sliders and live ROI projections.
"""

import os, pathlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Resolve paths relative to this script, works both locally and on Streamlit Cloud
APP_DIR = pathlib.Path(__file__).resolve().parent

from model.pipeline import (
    DEFAULT_OSINT, VALID_VALUES, FEATURE_COLS, ORDERED_SCALES,
    DEFAULT_SHIFT_RATES, clean_dataframe, engineer_features,
    get_distribution, shift_distribution, generate_future_cohort,
)

# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic AI ROI — Luxury Goods",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data & models (cached) ──────────────────────────────────────────

@st.cache_data
def load_data():
    df_synth_raw = pd.read_excel(APP_DIR / 'luxury_goods_synthetic_only_ai_positive.xlsx')
    df_synth = clean_dataframe(df_synth_raw)

    df_real_raw = pd.read_excel(APP_DIR / 'Luxury Goods Shopping Experince_March 11, 2026_17.48.xlsx')
    df_real = clean_dataframe(df_real_raw, drop_header_row=True)
    return df_synth, df_real

@st.cache_resource
def train_models(df_synth_raw):
    """Train models on the fly — avoids joblib version mismatch on cloud."""
    df = engineer_features(df_synth_raw)
    X, y = df[FEATURE_COLS], df['net_roi']
    models = {
        'XGBoost': XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=8,
                                                min_samples_split=5, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                         learning_rate=0.1, subsample=0.8, random_state=42),
    }
    for model in models.values():
        model.fit(X, y)
    return models

df_synth_raw, df_real_raw = load_data()
models = train_models(df_synth_raw)

# ── Sidebar: OSINT Parameter Controls ────────────────────────────────────

st.sidebar.markdown("# ⚙️ OSINT Parameters")
st.sidebar.markdown("Adjust luxury industry benchmarks from open-source intelligence.")

st.sidebar.markdown("### Customer Retention")
baseline_ret = st.sidebar.slider("Baseline Retention Rate", 0.50, 0.99, 0.82, 0.01,
                                  help="Avg luxury customer retention (Bain: ~82%)")
ai_ret_uplift = st.sidebar.slider("AI Retention Uplift", 0.05, 0.35, 0.15, 0.01,
                                   help="Max retention boost from AI personalization")
clv_mult = st.sidebar.slider("CLV Multiplier (x annual spend)", 3.0, 15.0, 8.5, 0.5,
                              help="Customer lifetime value as multiple of annual spend")
churn_cost = st.sidebar.slider("Churn Cost Multiplier", 2.0, 12.0, 5.0, 0.5,
                                help="Cost to acquire new customer vs retain existing")

st.sidebar.markdown("### AI Revenue Impact")
pers_lift = st.sidebar.slider("Personalization Revenue Lift", 0.05, 0.40, 0.20, 0.01,
                               help="Revenue increase from AI personalization")
ai_cost = st.sidebar.slider("AI Implementation Cost Ratio", 0.01, 0.15, 0.03, 0.005,
                              help="AI cost as fraction of customer revenue")

st.sidebar.markdown("### Projection Settings")
shift_rate = st.sidebar.slider("Annual AI Literacy Shift Rate", 0.05, 0.50, 0.25, 0.01,
                                help="How fast AI literacy grows per year (OSINT: ~25-30%)")
n_customers_proj = st.sidebar.slider("Projection Customer Base", 500, 10000, 1000, 500)
projection_end = st.sidebar.slider("Projection End Year", 2028, 2035, 2030)

osint = {
    'baseline_retention_rate': baseline_ret,
    'ai_enhanced_retention_uplift': ai_ret_uplift,
    'avg_luxury_clv_multiplier': clv_mult,
    'churn_cost_multiplier': churn_cost,
    'personalization_revenue_lift': pers_lift,
    'ai_implementation_cost_ratio': ai_cost,
    'income_tier_clv': DEFAULT_OSINT['income_tier_clv'],
    'purchase_frequency_weight': DEFAULT_OSINT['purchase_frequency_weight'],
    'age_digital_propensity': DEFAULT_OSINT['age_digital_propensity'],
}

shift_rates = {k: shift_rate for k in DEFAULT_SHIFT_RATES}

# ── Compute features with user params ────────────────────────────────────

df_synth = engineer_features(df_synth_raw, osint=osint)
df_real = engineer_features(df_real_raw, osint=osint)

# ── Header ────────────────────────────────────────────────────────────────

st.markdown("""
# 💎 Agentic AI ROI Dashboard — Luxury Goods Industry
**Interactive consulting for luxury brands: Agentic AI segmentation → use case match → ROI → roadmap.**
""")

# ── Sidebar navigation for closed-loop workflow ─────────────────────────────
page = st.sidebar.selectbox(
    "Navigation",
    ["Home Input", "Customer Segmentation", "Agentic AI Use Cases", "ROI Calculator", "Implementation Roadmap"],
)

# User input definitions
FEATURE_CHOICES = [
    "High-spending (High AOV) customers",
    "High-frequency repurchase customers",
    "Luxury limited/rare item buyers",
    "Omnichannel (online + offline) shoppers",
    "High private domain/community engagement customers",
    "Ultra-high-net-worth VIC/VIP clients",
    "At-risk churn customers (long time no purchase)",
    "Leather goods/jewelry/watch preference buyers",
]
PAIN_CHOICES = [
    "Insufficient personalized service for high-value clients",
    "Chaotic limited-edition product allocation & appointment",
    "Low customer repurchase & retention rate",
    "High customer service operational costs",
    "Inconsistent omnichannel customer experience",
    "Unfocused and inefficient luxury marketing",
]

def get_feature_df(df):
    # from engineered data in model.pipeline
    cols = [
        'annual_spend', 'purchase_freq_weight', 'brand_count', 'digital_propensity',
        'ai_usage_freq', 'clv', 'retention_improvement', 'satisfaction_score',
    ]
    return df[cols].fillna(0)

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

USE_CASE_RULES = [
    ("Ultra-High-Net-Worth Connoisseurs", "Insufficient personalized service for high-value clients", "AI Personal Luxury Concierge Agent", "Conversion +4%, AOV +8%, Labor -25%", 0.04, 0.08, 0.25),
    ("Luxury Collectors", "Chaotic limited-edition product allocation & appointment", "Limited-Edition Allocation & Pre-Release Agent", "Conversion +6%, AOV +15%, Labor -15%", 0.06, 0.15, 0.15),
    ("At-Risk Churn Customers", "Low customer repurchase & retention rate", "Customer Retention & Re-engagement Agent", "Conversion +2%, AOV +3%, Labor -10%", 0.02, 0.03, 0.10),
    (None, "High customer service operational costs", "Intelligent Private Domain Operation Agent", "Conversion +3%, AOV +5%, Labor -30%", 0.03, 0.05, 0.30),
    (None, "Inconsistent omnichannel customer experience", "Omnichannel Experience Synergy Agent", "Conversion +3%, AOV +5%, Labor -30%", 0.03, 0.05, 0.30),
]

if 'analysis' not in st.session_state:
    st.session_state.analysis = {}


def run_analysis(selected_features, selected_pains, annual_revenue):
    df = df_real.copy()
    feature_df = get_feature_df(df)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(feature_df)

    # Optimal cluster number via elbow method 3-5
    inertias = []
    for k in range(3, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X)
        inertias.append(km.inertia_)
    k_choices = [3, 4, 5]
    deltas = np.diff(inertias)
    if len(deltas) > 1:
        best_idx = int(np.argmin(deltas))
        n_clusters = k_choices[min(best_idx + 1, 2)]
    else:
        n_clusters = 3

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(X)
    df['cluster'] = kmeans.labels_

    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X)
    df['pc1'] = pca_coords[:, 0]
    df['pc2'] = pca_coords[:, 1]

    center_df = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=feature_df.columns)
    by_score = []

    for i, row in center_df.iterrows():
        score_atrisk = 1 - row['retention_improvement']
        score_uhnw = row['clv']
        score_collectors = row['brand_count'] + row['annual_spend']
        score_asp = row['ai_usage_freq'] + row['digital_propensity']
        score_gift = row['satisfaction_score']
        by_score.append((i, {
            'At-Risk Churn Customers': score_atrisk,
            'Ultra-High-Net-Worth Connoisseurs': score_uhnw,
            'Luxury Collectors': score_collectors,
            'Aspirational Luxury Buyers': score_asp,
            'Gift-Oriented Luxury Purchasers': score_gift,
        }))

    cluster_name_map = {}
    assigned = set()
    for cid, scores in sorted(by_score, key=lambda t: -max(t[1].values())):
        sorted_names = sorted(scores.items(), key=lambda x: -x[1])
        for name, _ in sorted_names:
            if name not in assigned:
                cluster_name_map[cid] = name
                assigned.add(name)
                break

    df['cluster_name'] = df['cluster'].map(cluster_name_map)

    # Use case matching
    matched = []
    for target_seg, pain, case_name, desc, conv, aov, labor in USE_CASE_RULES:
        seg_match = True if target_seg is None else target_seg in df['cluster_name'].unique()
        pain_match = pain in selected_pains
        if seg_match and pain_match:
            matched.append({
                'name': case_name,
                'description': desc,
                'conversion_lift': conv,
                'aov_lift': aov,
                'labor_saving': labor,
            })

    if not matched:
        # fallback default
        for _, pain, case_name, desc, conv, aov, labor in USE_CASE_RULES:
            if pain in selected_pains:
                matched.append({
                    'name': case_name,
                    'description': desc,
                    'conversion_lift': conv,
                    'aov_lift': aov,
                    'labor_saving': labor,
                })

    conversion_lift = sum(item['conversion_lift'] for item in matched)
    aov_lift = sum(item['aov_lift'] for item in matched)
    labor_saving = sum(item['labor_saving'] for item in matched)

    total_revenue_lift = annual_revenue * (conversion_lift + aov_lift)
    total_cost_saving = annual_revenue * 0.15 * labor_saving
    net_gain = total_revenue_lift + total_cost_saving
    ai_implementation_cost = net_gain * 0.10
    roi = (net_gain - ai_implementation_cost) / ai_implementation_cost if ai_implementation_cost > 0 else np.nan
    payback_months = ai_implementation_cost / net_gain * 12 if net_gain > 0 else np.nan

    # 3-year trend line forecast
    years = [1, 2, 3]
    trend = []
    base_cost = ai_implementation_cost
    for y in years:
        growth = 1 + 0.12 * (y - 1)
        trend.append({
            'year': f'Year {y}',
            'revenue_lift': total_revenue_lift * growth,
            'cost_saving': total_cost_saving * growth,
            'net_gain': net_gain * growth,
            'roi': roi,
        })

    roadmap_template = []
    if "Ultra-High-Net-Worth Connoisseurs" in df['cluster_name'].values:
        roadmap_template = [
            ("0-3 Months", "Data desensitization & integration; AI knowledge base; pilot concierge agent."),
            ("3-6 Months", "Limited-edition allocation agent launch; VIC private domain service scaling."),
            ("6-12 Months", "Full omnichannel synergy; multi-agent collaboration; ROI optimization."),
        ]
    elif "Aspirational Luxury Buyers" in df['cluster_name'].values or "Gift-Oriented Luxury Purchasers" in df['cluster_name'].values:
        roadmap_template = [
            ("0-3 Months", "Segmentation and personalization engine; scaled content & conversion workflow."),
            ("3-6 Months", "Targeted campaigns, digital experience upgrades, loyalty pipeline."),
            ("6-12 Months", "ROI scaling, repeat purchase programs, smart promotion automation."),
        ]
    else:
        roadmap_template = [
            ("0-3 Months", "Retention cohort tagging; re-engagement offer automation; special access triggers."),
            ("3-6 Months", "AI reactivation campaigns; concierge touchpoints; loyalty events."),
            ("6-12 Months", "Heatmap optimization; churn prediction; executive KPI review."),
        ]

    st.session_state.analysis = {
        'df': df,
        'cluster_name_map': cluster_name_map,
        'inertias': inertias,
        'kmeans_k': n_clusters,
        'pca': pca,
        'matched_use_cases': matched,
        'annual_revenue': annual_revenue,
        'conversion_lift': conversion_lift,
        'aov_lift': aov_lift,
        'labor_saving': labor_saving,
        'total_revenue_lift': total_revenue_lift,
        'total_cost_saving': total_cost_saving,
        'net_gain': net_gain,
        'ai_implementation_cost': ai_implementation_cost,
        'roi': roi,
        'payback_months': payback_months,
        'trend': trend,
        'roadmap': roadmap_template,
        'selected_features': selected_features,
        'selected_pains': selected_pains,
    }

# Home Input page
if page == "Home Input":
    st.header("Step 1: Customer Profile & Pain Point Input")
    st.markdown("""
    Provide luxury customer signal selections and business pain points. Annual revenue is used for ROI calibration.
    """)

    selected_features = st.multiselect("Customer Feature Selection", FEATURE_CHOICES)
    selected_pains = st.multiselect("Core Business Pain Points Selection", PAIN_CHOICES)
    annual_revenue = st.number_input("Brand Annual Sales/Revenue", min_value=0.0, value=5000000.0, step=10000.0, format="%.2f")

    if st.button("Run Closed-Loop Consulting Analysis"):
        if not selected_features:
            st.warning("Please select at least one customer feature.")
        elif not selected_pains:
            st.warning("Please select at least one business pain point.")
        else:
            run_analysis(selected_features, selected_pains, annual_revenue)
            st.success("Analysis complete. Navigate to other tabs via the sidebar.")

    if st.session_state.analysis:
        st.info("Analysis is ready. Navigate to Customer Segmentation, AI Use Cases, ROI Calculator and Implementation Roadmap.")
    st.stop()

# Customer Segmentation page
if page == "Customer Segmentation":
    st.header("Step 2: Customer Segmentation via K-Means")
    if not st.session_state.analysis:
        st.warning("Run the Home Input step first.")
        st.stop()

    analysis = st.session_state.analysis
    df = analysis['df']
    n_clusters = analysis['kmeans_k']

    st.markdown(f"Optimal cluster count estimated as **{n_clusters}** using elbow method (3-5).")
    cust_counts = df['cluster_name'].value_counts().reset_index()
    cust_counts.columns = ['Segment', 'Count']
    st.table(cust_counts)

    fig_pca = px.scatter(df, x='pc1', y='pc2', color='cluster_name', size='clv',
                         title='2D PCA Cluster Distribution', width=900, height=500)
    st.plotly_chart(fig_pca, use_container_width=True)

    # Radar per cluster
    radar_data = []
    for seg in df['cluster_name'].unique():
        seg_df = df[df['cluster_name'] == seg]
        radar_data.append({
            'segment': seg,
            'annual_spend': seg_df['annual_spend'].mean(),
            'purchase_freq_weight': seg_df['purchase_freq_weight'].mean(),
            'clv': seg_df['clv'].mean(),
            'ai_usage_freq': seg_df['ai_usage_freq'].mean(),
            'retention_improvement': seg_df['retention_improvement'].mean(),
        })

    radar_df = pd.DataFrame(radar_data)
    if not radar_df.empty:
        categories = ['annual_spend', 'purchase_freq_weight', 'clv', 'ai_usage_freq', 'retention_improvement']
        fig = go.Figure()
        for _, row in radar_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=row[categories].values,
                theta=[c.replace('_', ' ').title() for c in categories],
                fill='toself',
                name=row['segment'],
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, radar_df[categories].max().max() * 1.1]),
                angularaxis=dict(direction='clockwise')
            ),
            showlegend=True,
            title='Cluster Feature Radar'
        )
        st.plotly_chart(fig, use_container_width=True)

    pie = px.pie(cust_counts, names='Segment', values='Count', title='Segment Proportion')
    st.plotly_chart(pie, use_container_width=True)
    st.stop()

# Agentic AI Use Cases page
if page == "Agentic AI Use Cases":
    st.header("Step 3: Agentic AI Use Case Matching")
    if not st.session_state.analysis:
        st.warning("Run the Home Input step first.")
        st.stop()

    matched = st.session_state.analysis['matched_use_cases']
    if not matched:
        st.info("No use cases matched. Please refine pain points or segmentation.")
        st.stop()

    cols = st.columns(2)
    for i, uc in enumerate(matched):
        with cols[i % 2]:
            st.markdown(f"### {uc['name']}")
            st.markdown(f"**Benefits:** {uc['description']}")
            st.markdown(f"- Conversion lift: {uc['conversion_lift']*100:.1f}%")
            st.markdown(f"- AOV lift: {uc['aov_lift']*100:.1f}%")
            st.markdown(f"- Labor saving: {uc['labor_saving']*100:.1f}%")
    st.stop()

# ROI Calculator page
if page == "ROI Calculator":
    st.header("Step 4: Customized ROI Calculation")
    if not st.session_state.analysis:
        st.warning("Run the Home Input step first.")
        st.stop()

    a = st.session_state.analysis
    st.metric("Total Revenue Lift", f"${a['total_revenue_lift']:,.0f}")
    st.metric("Total Cost Saving", f"${a['total_cost_saving']:,.0f}")
    st.metric("Net Annual Gain", f"${a['net_gain']:,.0f}")
    st.metric("AI Implementation Cost", f"${a['ai_implementation_cost']:,.0f}")
    st.metric("ROI", f"{a['roi']:.2f}x")
    st.metric("Payback Period", f"{a['payback_months']:.1f} months")

    dfk = pd.DataFrame(a['trend'])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dfk['year'], y=dfk['revenue_lift'], name='Revenue Lift'))
    fig.add_trace(go.Bar(x=dfk['year'], y=dfk['cost_saving'], name='Cost Saving'))
    fig.add_trace(go.Line(x=dfk['year'], y=dfk['net_gain'], name='Net Gain'))
    fig.update_layout(title='3-Year ROI Trend', yaxis_title='USD', width=900, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.stop()

# Implementation Roadmap page
if page == "Implementation Roadmap":
    st.header("Step 5: Auto-Generated Implementation Roadmap")
    if not st.session_state.analysis:
        st.warning("Run the Home Input step first.")
        st.stop()

    roadmap = st.session_state.analysis['roadmap']
    st.table(pd.DataFrame(roadmap, columns=['Phase', 'Actions']))
    st.markdown("### Risk Assessment & Mitigation")
    st.markdown("- Data privacy risk: ensure strict desensitization and consent management.")
    st.markdown("- Allocation risk: pilot limited-edition agent with controlled cohorts.")
    st.markdown("- Retention risk: monitor at-risk churn signals weekly and trigger re-engagement.")
    st.markdown("- Omnichannel risk: unify CRM and POS data streams before full deployment.")
    st.stop()

# End of new navigation pages

# (Legacy tabs removed; new sidebar workflow drives experience.)
# =====================================================================

with tab1:
    st.header("Current State Analysis")

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Net ROI", f"{df_real['net_roi'].mean():.2f}x")
    c2.metric("Median Net ROI", f"{df_real['net_roi'].median():.2f}x")
    c3.metric("Avg AI Readiness", f"{df_real['ai_readiness'].mean():.3f}")
    c4.metric("Retention Uplift", f"+{df_real['retention_improvement'].mean():.1%}")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("ROI Distribution — Real vs Synthetic")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df_synth['net_roi'], name=f"Synthetic (n={len(df_synth)})",
                                    opacity=0.5, nbinsx=25, marker_color='#3498db'))
        fig.add_trace(go.Histogram(x=df_real['net_roi'], name=f"Real (n={len(df_real)})",
                                    opacity=0.7, nbinsx=15, marker_color='#e74c3c'))
        fig.update_layout(barmode='overlay', xaxis_title='Net ROI (x)', yaxis_title='Count',
                          height=400, margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("ROI vs AI Readiness")
        fig = px.scatter(df_real, x='ai_readiness', y='net_roi', color='annual_spend',
                         color_continuous_scale='YlOrRd', size_max=12,
                         labels={'ai_readiness': 'AI Readiness', 'net_roi': 'Net ROI (x)',
                                 'annual_spend': 'Annual Spend ($)'})
        fig.update_layout(height=400, margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

    # Model metrics
    st.subheader("Model Performance")
    X_real = df_real[FEATURE_COLS]
    y_real = df_real['net_roi']

    perf_rows = []
    for name, model in models.items():
        y_pred = model.predict(X_real)
        r2 = r2_score(y_real, y_pred)
        mae = mean_absolute_error(y_real, y_pred)
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        perf_rows.append({'Model': name, 'R²': round(r2, 4), 'MAE': round(mae, 4), 'RMSE': round(rmse, 4)})

    st.dataframe(pd.DataFrame(perf_rows).set_index('Model'), use_container_width=True)

    # Feature importance
    st.subheader("Feature Importance (Gradient Boosting)")
    gb = models['Gradient Boosting']
    feat_imp = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Importance': gb.feature_importances_,
    }).sort_values('Importance', ascending=True)

    fig = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                 color='Importance', color_continuous_scale='Greens')
    fig.update_layout(height=500, margin=dict(t=10, b=30), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# TAB 2: FORWARD PROJECTION
# =====================================================================

with tab2:
    st.header(f"AI Literacy → ROI Projection (2026–{projection_end})")
    st.markdown(f"""
    *Generating **500 synthetic respondents per year** with AI literacy distributions
    shifted at **{shift_rate:.0%}/year** based on OSINT trends.*
    """)

    # Build baseline distributions from real data
    baseline_dists = {}
    for col in ORDERED_SCALES:
        valid = set(ORDERED_SCALES[col])
        if col == 'Q16':
            valid = {'yes', 'Yes', 'No', 'no'}
        baseline_dists[col] = get_distribution(df_real_raw.get(col, pd.Series(dtype=str)), valid)

    demo_dists = {}
    for col in ['Q1', 'Q2', 'Q3', 'Q4', 'Q7', 'Q11']:
        demo_dists[col] = get_distribution(df_real_raw.get(col, pd.Series(dtype=str)), VALID_VALUES[col])

    multi_cols = ['Q5', 'Q15', 'Q20', 'Q23']
    existing_multi = [c for c in multi_cols if c in df_real_raw.columns]
    multi_df = df_real_raw[existing_multi].dropna(how='all') if existing_multi else None

    years = list(range(2026, projection_end + 1))
    proj_data = []

    for year in years:
        df_y = generate_future_cohort(
            baseline_dists, demo_dists, multi_df,
            year=year, shift_rates=shift_rates, osint=osint,
        )
        y_pred = models['Gradient Boosting'].predict(df_y[FEATURE_COLS])
        proj_data.append({
            'Year': year,
            'AI Readiness': df_y['ai_readiness'].mean(),
            'AI-Ready (>0.5)': (df_y['ai_readiness'] > 0.5).mean(),
            'Formula ROI': df_y['net_roi'].mean(),
            'Model ROI': y_pred.mean(),
            'Revenue Uplift': df_y['ai_revenue_uplift'].mean(),
            'Retention Savings': df_y['retention_savings'].mean(),
            'AI Cost': df_y['ai_cost'].mean(),
            'Net Benefit': (df_y['ai_revenue_uplift'].mean()
                            + df_y['retention_savings'].mean()
                            - df_y['ai_cost'].mean()),
            'Retention Improvement': df_y['retention_improvement'].mean(),
            'High ROI (>1x)': (df_y['net_roi'] > 1.0).mean(),
            'df': df_y,
        })

    proj_df = pd.DataFrame(proj_data)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    last = proj_df.iloc[-1]
    first = proj_df.iloc[0]
    c1.metric(f"ROI in {projection_end}", f"{last['Formula ROI']:.2f}x",
              delta=f"+{last['Formula ROI'] - first['Formula ROI']:.2f}x from 2026")
    c2.metric("AI-Ready Customers", f"{last['AI-Ready (>0.5)']:.0%}",
              delta=f"+{last['AI-Ready (>0.5)'] - first['AI-Ready (>0.5)']:.0%}")
    cumulative_net = proj_df['Net Benefit'].sum() * n_customers_proj
    c3.metric(f"Cumulative Benefit ({len(years)}yr)", f"${cumulative_net/1e6:.1f}M",
              help=f"Per {n_customers_proj:,} customers")
    c4.metric("Retention by End", f"{baseline_ret + last['Retention Improvement']:.1%}",
              delta=f"+{last['Retention Improvement']:.1%}")

    # ROI trajectory chart
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("ROI Trajectory")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=proj_df['Year'], y=proj_df['Formula ROI'], mode='lines+markers',
                                  name='Formula ROI', line=dict(color='#e74c3c', width=3), marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=proj_df['Year'], y=proj_df['Model ROI'], mode='lines+markers',
                                  name='Model Predicted ROI', line=dict(color='#3498db', width=3, dash='dash'),
                                  marker=dict(size=10)))
        fig.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="Breakeven (1.0x)")
        fig.update_layout(xaxis_title='Year', yaxis_title='Mean Net ROI (x)',
                          height=400, margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("AI Readiness Growth")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=proj_df['Year'], y=proj_df['AI-Ready (>0.5)'] * 100,
                              name='% AI-Ready', marker_color='rgba(52,152,219,0.3)'), secondary_y=False)
        fig.add_trace(go.Scatter(x=proj_df['Year'], y=proj_df['AI Readiness'], mode='lines+markers',
                                  name='Mean Readiness', line=dict(color='#e74c3c', width=3),
                                  marker=dict(size=10)), secondary_y=True)
        fig.update_yaxes(title_text="% Customers AI-Ready", secondary_y=False)
        fig.update_yaxes(title_text="Mean AI Readiness Score", secondary_y=True)
        fig.update_layout(xaxis_title='Year', height=400, margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

    # Value components stacked area
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.subheader("Value Components Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=proj_df['Year'], y=proj_df['Revenue Uplift'],
                                  fill='tozeroy', name='Revenue Uplift', line=dict(color='#27ae60')))
        fig.add_trace(go.Scatter(x=proj_df['Year'], y=proj_df['Revenue Uplift'] + proj_df['Retention Savings'],
                                  fill='tonexty', name='+ Retention Savings', line=dict(color='#2ecc71')))
        fig.add_trace(go.Scatter(x=proj_df['Year'], y=proj_df['AI Cost'], mode='lines+markers',
                                  name='AI Cost', line=dict(color='#e74c3c', width=2, dash='dash')))
        fig.update_layout(xaxis_title='Year', yaxis_title='$ per Customer',
                          height=400, margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        st.subheader(f"Cumulative Net Benefit (per {n_customers_proj:,} customers)")
        net_per_yr = proj_df['Net Benefit'] * n_customers_proj
        cum = net_per_yr.cumsum()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=proj_df['Year'], y=net_per_yr / 1e6,
                              name='Annual', marker_color='rgba(52,152,219,0.6)'))
        fig.add_trace(go.Scatter(x=proj_df['Year'], y=cum / 1e6, mode='lines+markers',
                                  name='Cumulative', line=dict(color='#e74c3c', width=3),
                                  marker=dict(size=10)))
        fig.update_layout(xaxis_title='Year', yaxis_title='$ Millions',
                          height=400, margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

    # AI literacy distribution shift
    st.subheader("AI Usage Frequency — Distribution Shift")
    scale_q13 = ORDERED_SCALES['Q13']
    shift_data = []
    for year in years:
        dist = shift_distribution(baseline_dists.get('Q13', {}), scale_q13,
                                   shift_rates.get('Q13', shift_rate), year - 2026)
        for level, pct in dist.items():
            shift_data.append({'Year': str(year), 'Level': level, 'Percentage': pct * 100})
    shift_df = pd.DataFrame(shift_data)
    fig = px.bar(shift_df, x='Year', y='Percentage', color='Level', barmode='stack',
                 color_discrete_map={'Never': '#e74c3c', 'Rarely': '#f39c12',
                                      'Occasionally': '#3498db', 'Yes, frequently': '#27ae60'})
    fig.update_layout(yaxis_title='% of Customers', height=400, margin=dict(t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # Projection data table
    with st.expander("View Projection Data Table"):
        display_cols = [c for c in proj_df.columns if c != 'df']
        st.dataframe(proj_df[display_cols].set_index('Year').round(3), use_container_width=True)


# =====================================================================
# TAB 3: SEGMENT DEEP DIVE
# =====================================================================

with tab3:
    st.header("Segment Deep Dive")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("ROI by Income Tier")
        income_roi = df_real.groupby('Q3').agg(
            mean_roi=('net_roi', 'mean'), mean_clv=('clv', 'mean'),
            count=('net_roi', 'count'),
        ).sort_values('mean_roi', ascending=True).reset_index()
        fig = px.bar(income_roi, x='mean_roi', y='Q3', orientation='h',
                     color='mean_roi', color_continuous_scale='RdYlGn',
                     text=income_roi['mean_roi'].apply(lambda v: f'{v:.2f}x'),
                     labels={'Q3': 'Income Tier', 'mean_roi': 'Mean ROI'})
        fig.update_layout(height=400, margin=dict(t=10, b=30), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("ROI by Age Group")
        age_order = ['Under 18', '18–24', '25–34', '35–44', '45–54', '55–64', '65+']
        age_roi = df_real.groupby('Q1').agg(
            mean_roi=('net_roi', 'mean'),
            mean_ai_readiness=('ai_readiness', 'mean'),
            count=('net_roi', 'count'),
        ).reindex(age_order).dropna().reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=age_roi['Q1'], y=age_roi['mean_roi'],
                              name='Mean ROI', marker_color='#3498db'), secondary_y=False)
        fig.add_trace(go.Scatter(x=age_roi['Q1'], y=age_roi['mean_ai_readiness'],
                                  mode='lines+markers', name='AI Readiness',
                                  line=dict(color='#e74c3c', width=2)), secondary_y=True)
        fig.update_yaxes(title_text="Mean ROI (x)", secondary_y=False)
        fig.update_yaxes(title_text="AI Readiness", secondary_y=True)
        fig.update_layout(height=400, margin=dict(t=10, b=30))
        st.plotly_chart(fig, use_container_width=True)

    # AI adoption level
    st.subheader("ROI by AI Adoption Level")
    df_real_copy = df_real.copy()
    df_real_copy['AI Adoption'] = pd.cut(df_real_copy['ai_readiness'],
                                          bins=[0, 0.25, 0.5, 0.75, 1.0],
                                          labels=['Low', 'Medium', 'High', 'Very High'])
    adopt_roi = df_real_copy.groupby('AI Adoption', observed=True).agg(
        mean_roi=('net_roi', 'mean'),
        revenue_uplift=('ai_revenue_uplift', 'mean'),
        retention_savings=('retention_savings', 'mean'),
        count=('net_roi', 'count'),
    ).reset_index()

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        fig = px.bar(adopt_roi, x='AI Adoption', y='mean_roi', color='mean_roi',
                     color_continuous_scale='RdYlGn',
                     text=adopt_roi['mean_roi'].apply(lambda v: f'{v:.2f}x'),
                     labels={'mean_roi': 'Mean ROI'})
        fig.update_layout(height=350, margin=dict(t=10, b=30), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=adopt_roi['AI Adoption'], y=adopt_roi['revenue_uplift'],
                              name='Revenue Uplift', marker_color='#27ae60'))
        fig.add_trace(go.Bar(x=adopt_roi['AI Adoption'], y=adopt_roi['retention_savings'],
                              name='Retention Savings', marker_color='#2ecc71'))
        fig.update_layout(barmode='group', yaxis_title='$ per Customer',
                          height=350, margin=dict(t=10, b=30))
        st.plotly_chart(fig, use_container_width=True)

    # Retention uplift by age
    st.subheader("AI-Driven Retention Uplift by Age")
    ret_by_age = df_real.groupby('Q1')['retention_improvement'].mean().reindex(age_order).dropna().reset_index()
    ret_by_age.columns = ['Age Group', 'Retention Improvement']
    fig = px.bar(ret_by_age, x='Age Group', y=ret_by_age['Retention Improvement'] * 100,
                 color=ret_by_age['Retention Improvement'] * 100, color_continuous_scale='Blues',
                 text=ret_by_age['Retention Improvement'].apply(lambda v: f'{v*100:.1f}%'),
                 labels={'y': 'Retention Improvement (%)'})
    fig.update_layout(height=350, margin=dict(t=10, b=30), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("Feature Correlation Matrix")
    corr_cols = ['ai_readiness', 'satisfaction_score', 'ai_helpfulness',
                 'ai_desire', 'retention_improvement', 'net_roi', 'clv']
    corr = df_real[corr_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1, aspect='auto')
    fig.update_layout(height=450, margin=dict(t=10, b=30))
    st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# TAB 4: EXECUTIVE SUMMARY
# =====================================================================

with tab4:
    st.header("Executive Summary")

    # Build metrics for summary
    last_yr = proj_df.iloc[-1]
    first_yr = proj_df.iloc[0]
    cum_benefit = proj_df['Net Benefit'].sum() * n_customers_proj

    st.markdown(f"""
    ### Model Performance
    - **Best Model:** Gradient Boosting (Transfer R² on real data)
    - **Training Data:** {len(df_synth)} synthetic survey respondents
    - **Real Evaluation Data:** {len(df_real)} real survey respondents

    ### Current State (2026 — Real Survey)
    | Metric | Value |
    |--------|-------|
    | Mean Net ROI | **{df_real['net_roi'].mean():.2f}x** |
    | Median Net ROI | {df_real['net_roi'].median():.2f}x |
    | AI Readiness (mean) | {df_real['ai_readiness'].mean():.3f} |
    | AI-Ready Customers (>0.5) | {(df_real['ai_readiness'] > 0.5).mean():.0%} |
    | Baseline Retention | {baseline_ret:.0%} |
    | Retention Improvement | +{df_real['retention_improvement'].mean():.1%} |
    | Revenue Uplift / Customer | ${df_real['ai_revenue_uplift'].mean():,.0f} |
    | Retention Savings / Customer | ${df_real['retention_savings'].mean():,.0f} |
    | AI Cost / Customer | ${df_real['ai_cost'].mean():,.0f} |

    ### {len(years)}-Year Projection (2026–{projection_end})
    | Metric | 2026 | {projection_end} | Change |
    |--------|------|------|--------|
    | Mean ROI | {first_yr['Formula ROI']:.2f}x | **{last_yr['Formula ROI']:.2f}x** | +{last_yr['Formula ROI'] - first_yr['Formula ROI']:.2f}x |
    | AI Readiness | {first_yr['AI Readiness']:.3f} | {last_yr['AI Readiness']:.3f} | +{(last_yr['AI Readiness']/first_yr['AI Readiness'] - 1):.0%} |
    | AI-Ready (>0.5) | {first_yr['AI-Ready (>0.5)']:.0%} | {last_yr['AI-Ready (>0.5)']:.0%} | +{last_yr['AI-Ready (>0.5)'] - first_yr['AI-Ready (>0.5)']:.0%} |
    | Post-AI Retention | {baseline_ret + first_yr['Retention Improvement']:.1%} | {baseline_ret + last_yr['Retention Improvement']:.1%} | +{last_yr['Retention Improvement'] - first_yr['Retention Improvement']:.1%} |

    ### Financial Impact (per {n_customers_proj:,} customers)
    | Year | Revenue Uplift | Retention Savings | AI Cost | Net Benefit |
    |------|---------------|-------------------|---------|-------------|
    """)

    for _, row in proj_df.iterrows():
        nb = row['Net Benefit'] * n_customers_proj
        st.markdown(
            f"| {int(row['Year'])} | ${row['Revenue Uplift']*n_customers_proj:,.0f} | "
            f"${row['Retention Savings']*n_customers_proj:,.0f} | "
            f"${row['AI Cost']*n_customers_proj:,.0f} | **${nb:,.0f}** |"
        )

    st.markdown(f"""
    | **Cumulative** | | | | **${cum_benefit:,.0f}** |

    ### OSINT Parameters Used
    | Parameter | Value |
    |-----------|-------|
    | Baseline Retention Rate | {baseline_ret:.0%} |
    | AI Retention Uplift | {ai_ret_uplift:.0%} |
    | CLV Multiplier | {clv_mult}x |
    | Churn Cost Multiplier | {churn_cost}x |
    | Personalization Revenue Lift | {pers_lift:.0%} |
    | AI Implementation Cost Ratio | {ai_cost:.1%} |
    | Annual AI Literacy Shift Rate | {shift_rate:.0%} |

    ### Recommendation
    Deploying agentic AI in luxury retail yields a projected **{last_yr['Formula ROI']:.1f}x ROI by {projection_end}**,
    growing from {first_yr['Formula ROI']:.2f}x in 2026. The **{len(years)}-year cumulative net benefit of
    ${cum_benefit/1e6:.1f}M per {n_customers_proj:,} customers** makes a strong case for phased deployment
    starting in 2026, with accelerated rollout as adoption crosses the inflection point.
    Priority segments are **high-income, digitally-native customers (25–34)** with existing AI familiarity.
    """)

    # Download button
    csv_data = proj_df[[c for c in proj_df.columns if c != 'df']].to_csv(index=False)
    st.download_button("📥 Download Projection Data (CSV)", csv_data,
                       "roi_projection.csv", "text/csv")
