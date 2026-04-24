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
from bedrock_agent import (
    AI_USE_CASES, TEAM_SIZE_OPTIONS, CONSTRAINT_OPTIONS,
    build_prompt, invoke_bedrock, render_roadmap,
    search_osint, format_osint_context, validate_inputs,
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
**Interactive analysis of AI deployment ROI, powered by survey data and OSINT retention benchmarks.**
""")

# ── Tabs ──────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Current State", "🚀 Forward Projection", "🔍 Segment Deep Dive",
    "📋 Executive Summary", "🤖 AI Implementation Advisor",
])

# =====================================================================
# TAB 1: CURRENT STATE
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


# =====================================================================
# TAB 5: AI IMPLEMENTATION ADVISOR
# =====================================================================

with tab5:
    st.header("AI Implementation Advisor")
    st.markdown("""
    *Get a tailored implementation roadmap based on your
    AI use case, budget, timeline, and organizational constraints.*
    """)

    with st.form("advisor_form"):
        col_a, col_b = st.columns(2)

        with col_a:
            selected_use_case = st.selectbox(
                "Select AI Use Case",
                options=list(AI_USE_CASES.keys()),
                help="Choose the primary AI capability you want to deploy.",
            )
            st.caption(AI_USE_CASES[selected_use_case])

            budget = st.number_input(
                "Total Budget ($)", min_value=10_000, max_value=50_000_000,
                value=500_000, step=50_000, help="Total budget for the AI initiative.",
            )
            timeline = st.slider(
                "Timeline (months)", min_value=3, max_value=36, value=12,
                help="Desired implementation timeline.",
            )

        with col_b:
            team_size = st.selectbox("Current Team Size", TEAM_SIZE_OPTIONS)
            constraints = st.multiselect(
                "Key Constraints", CONSTRAINT_OPTIONS,
                default=["Must maintain luxury brand experience"],
                help="Select all that apply.",
            )
            additional = st.text_area(
                "Additional Details (optional)",
                placeholder="Describe specific requirements, existing tech stack, "
                            "target customer segments, etc.",
                height=120,
            )

        submitted = st.form_submit_button("Generate Implementation Roadmap",
                                           type="primary", use_container_width=True)

    if submitted:
        input_warnings = validate_inputs(budget, timeline, team_size, constraints)
        if input_warnings:
            for w in input_warnings:
                st.warning(w)

        roi_context = {
            "mean_roi": round(df_real['net_roi'].mean(), 2),
            "ai_ready_pct": f"{(df_real['ai_readiness'] > 0.5).mean():.0%}",
            "avg_readiness": round(df_real['ai_readiness'].mean(), 3),
            "avg_revenue_uplift": df_real['ai_revenue_uplift'].mean(),
            "avg_retention_savings": df_real['retention_savings'].mean(),
        }

        with st.spinner("Searching for latest industry data (OSINT)..."):
            osint_results = search_osint(selected_use_case)
            osint_ctx = format_osint_context(osint_results)
            if osint_results:
                st.session_state["advisor_osint"] = osint_results
            else:
                st.session_state.pop("advisor_osint", None)

        prompt = build_prompt(
            use_case=selected_use_case,
            use_case_desc=AI_USE_CASES[selected_use_case],
            budget=budget,
            timeline_months=timeline,
            team_size=team_size,
            constraints=constraints,
            additional_details=additional,
            current_roi_data=roi_context,
            osint_context=osint_ctx,
        )

        with st.spinner("Generating your AI implementation roadmap..."):
            try:
                roadmap = invoke_bedrock(prompt)
                st.session_state["advisor_roadmap"] = roadmap
                st.session_state.pop("advisor_downloads", None)
            except Exception as e:
                import traceback as tb, sys
                print(f"[BEDROCK ERROR] {type(e).__name__}: {e}", file=sys.stderr)
                tb.print_exc(file=sys.stderr)
                st.session_state.pop("advisor_roadmap", None)
                error_msg = str(e)
                if "NoCredentialsError" in error_msg or "credentials" in error_msg.lower():
                    st.error(
                        "**Service configuration issue.** "
                        "Please contact the administrator to verify the "
                        "deployment credentials are properly configured."
                    )
                elif "AccessDeniedException" in error_msg:
                    st.error(
                        "**Access denied.** The AI service permissions "
                        "need to be configured by the administrator."
                    )
                else:
                    st.error(f"Error generating roadmap: {error_msg}")

    if "advisor_roadmap" in st.session_state:
        st.markdown("---")
        render_roadmap(st.session_state["advisor_roadmap"])
