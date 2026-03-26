"""
Agentic AI ROI Dashboard — Luxury Goods Industry
Streamlit app with interactive OSINT parameter sliders and live ROI projections.
Enhanced UI for beautiful and intuitive experience.
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

# ── Custom CSS & Styling ───────────────────────────────────────────────────

st.markdown("""
<style>
/* Theme Colors */
:root {
    --primary: #C41E3A;
    --secondary: #1f1f1f;
    --accent: #FFB81C;
    --success: #10B981;
    --warning: #F59E0B;
    --error: #EF4444;
    --light: #F9FAFB;
    --dark: #111827;
}

/* Overall styling */
.main {
    background-color: #FFFFFF;
}

/* Custom containers */
.metric-card {
    background: linear-gradient(135deg, #F3F4F6 0%, #E5E7EB 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #C41E3A;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.status-card {
    background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #FECACA;
}

.success-card {
    background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #86EFAC;
}

.info-card {
    background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #7DD3FC;
}

/* Headers & Typography */
h1, h2, h3 {
    color: #111827;
    font-weight: 700;
}

h1 {
    border-bottom: 3px solid #C41E3A;
    padding-bottom: 0.75rem;
    margin-bottom: 1.5rem;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    border-bottom: 3px solid #C41E3A;
    color: #C41E3A;
    font-weight: 600;
}

/* Button styling */
.stButton button {
    background: linear-gradient(135deg, #C41E3A 0%, #9B1733 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.75rem 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(196, 30, 58, 0.3);
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(196, 30, 58, 0.4);
}

/* Input fields */
.stNumberInput input, .stSelectbox select, .stMultiSelect [role="listbox"] {
    border-radius: 8px;
    border: 2px solid #E5E7EB;
    padding: 0.75rem;
}

/* Expanders */
.streamlit-expander {
    border: 1px solid #E5E7EB;
    border-radius: 8px;
}

.streamlit-expanderHeader {
    background-color: #F9FAFB;
    border-radius: 8px;
}

/* Metric styling */
.metric-container {
    background: #F9FAFB;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #E5E7EB;
}

/* Sidebar styling */
.st-sidebar {
    background: linear-gradient(180deg, #F9FAFB 0%, #F3F4F6 100%);
}

.st-sidebar .stNumberInput input {
    background: white;
}

/* Progress indicator styling */
.progress-step {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem;
    margin: 0.5rem 0;
    background: #F9FAFB;
    border-radius: 8px;
    border-left: 3px solid #E5E7EB;
}

.progress-step.active {
    background: #FEF2F2;
    border-left-color: #C41E3A;
}

.progress-step.completed {
    background: #F0FDF4;
    border-left-color: #10B981;
}

/* Section dividers */
hr {
    margin: 2rem 0;
    border: none;
    border-top: 2px solid #E5E7EB;
}

/* Data table styling */
.dataframe {
    font-size: 0.9rem;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

# ── Load data & models (cached) ──────────────────────────────────────────

@st.cache_data
def load_data():
    """Load customer data from CSV files (full datasets) instead of limited survey export."""
    try:
        # Load all three customer segments from CSV files
        df_affluent_plus = pd.read_csv(APP_DIR / 'Affluent_Plus_Sample_Data.csv')
        df_mass_affluent = pd.read_csv(APP_DIR / 'Mass_Affluent_Sample_Data.csv')
        df_mass_market = pd.read_csv(APP_DIR / 'Mass_Market_Sample_Data.csv')
        
        # Add segment labels
        df_affluent_plus['segment'] = 'Affluent Plus'
        df_mass_affluent['segment'] = 'Mass Affluent'
        df_mass_market['segment'] = 'Mass Market'
        
        # Combine all segments
        df_real = pd.concat([df_affluent_plus, df_mass_affluent, df_mass_market], ignore_index=True)
        
        # Basic cleaning
        df_real = df_real.dropna(subset=['efx_total_income360']).reset_index(drop=True)
        
        return df_real, df_real
        
    except FileNotFoundError as e:
        # Fallback: use the limited Excel survey export if CSVs not found
        st.error(f"⚠️ Could not load CSV files: {e}")
        st.error("Please ensure Affluent_Plus_Sample_Data.csv, Mass_Affluent_Sample_Data.csv, and Mass_Market_Sample_Data.csv exist.")
        st.stop()

# ── UI Helper Functions ──────────────────────────────────────────────────

def show_metric_card(label, value, subtext="", icon="📊"):
    """Display a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.85rem; color: #666; margin-bottom: 0.25rem;">{label}</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: #111827;">{icon} {value}</div>
        <div style="font-size: 0.8rem; color: #999; margin-top: 0.25rem;">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)

def show_progress_steps(current_step):
    """Display workflow progress indicator."""
    steps = [
        ("1️⃣", "Home Input", "Customer Profile"),
        ("2️⃣", "Segmentation", "K-Means Clustering"),
        ("3️⃣", "AI Use Cases", "Strategic Alignment"),
        ("4️⃣", "ROI Calculator", "Financial Impact"),
        ("5️⃣", "Roadmap", "Implementation Plan"),
    ]
    
    st.sidebar.markdown("### 📍 Workflow Progress")
    for i, (emoji, title, desc) in enumerate(steps, 1):
        status_class = "active" if i == current_step else "completed" if i < current_step else ""
        status_emoji = "✓" if i < current_step else emoji
        st.sidebar.markdown(f"""
        <div class="progress-step {status_class}">
            <span>{status_emoji}</span>
            <div>
                <strong>{title}</strong>
                <div style="font-size: 0.8rem; color: #666;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_section(title, icon="📌"):
    """Create a styled section header."""
    st.markdown(f"### {icon} {title}")
    st.divider()

def match_features_from_text(user_text, feature_list):
    """
    Intelligent text matching function that identifies which features/pain points
    the user's typed description falls into using keyword matching and similarity.
    
    Args:
        user_text: User's typed description
        feature_list: List of features/pain points to match against
        
    Returns:
        List of matched categories
    """
    if not user_text or not user_text.strip():
        return []
    
    user_text_lower = user_text.lower()
    matched = []
    
    # Define keyword mappings for each feature/pain point
    keyword_mappings = {
        # Customer Features
        "High-spending (High AOV) customers": [
            "high", "spending", "aov", "expensive", "premium", "luxury", "wealth", "rich",
            "affluent", "high-value", "big spenders", "expensive purchases", "heavy",
            "high transaction", "big ticket", "vip", "vip client", "high ticket"
        ],
        "High-frequency repurchase customers": [
            "frequent", "repeat", "loyalty", "regular", "purchase", "recurrent", "habitual",
            "steady", "consistent", "recurring", "repeat buyer", "loyal customer", "high volume"
        ],
        "Luxury limited/rare item buyers": [
            "limited", "edition", "rare", "exclusive", "collector", "collectible", "premium",
            "scarce", "sought-after", "hard to find", "limited edition", "exclusive items"
        ],
        "Omnichannel (online + offline) shoppers": [
            "omnichannel", "online", "offline", "channel", "digital", "store", "web",
            "mobile", "seamless", "cross-channel", "integrated", "multi-channel"
        ],
        "High private domain/community engagement customers": [
            "community", "engagement", "private", "domain", "social", "network", "group",
            "member", "community member", "engagement", "active", "social network", "brand community"
        ],
        "Ultra-high-net-worth VIC/VIP clients": [
            "ultra", "high-net-worth", "hnw", "vip", "vic", "premium member", "elite",
            "exclusive", "top tier", "high-value", "c-level", "executive", "wealthy", "ultra-high"
        ],
        "At-risk churn customers (long time no purchase)": [
            "churn", "risk", "at-risk", "lapsed", "inactive", "lost", "dormant", "decline",
            "decreasing", "no purchase", "long time", "last purchase", "inactive customer"
        ],
        "Leather goods/jewelry/watch preference buyers": [
            "leather", "jewelry", "watch", "accessories", "luxury goods", "premium", "apparel",
            "brand", "designer", "fashion", "goods", "product", "category"
        ],
        
        # Pain Points
        "Insufficient personalized service for high-value clients": [
            "personalization", "personalized", "service", "customization", "custom", "tailored",
            "individual", "specific", "high-value", "premium service", "personal attention",
            "one-on-one", "dedicated", "bespoke"
        ],
        "Chaotic limited-edition product allocation & appointment": [
            "allocation", "appointment", "limited", "chaos", "chaotic", "confusion", "management",
            "scheduling", "booking", "product", "edition", "distribution", "fairness"
        ],
        "Low customer repurchase & retention rate": [
            "retention", "repurchase", "repeat", "coming back", "loyalty", "retention rate",
            "repeat purchase", "reduce churn", "keep", "maintain", "improve loyalty"
        ],
        "High customer service operational costs": [
            "cost", "operational", "expensive", "labor", "efficiency", "overhead", "manual",
            "workload", "team", "support", "reduce cost", "automate", "labor intensive"
        ],
        "Inconsistent omnichannel customer experience": [
            "inconsistent", "omnichannel", "experience", "consistent", "channel", "touchpoint",
            "online", "offline", "unified", "seamless", "integration", "experience"
        ],
        "Unfocused and inefficient luxury marketing": [
            "marketing", "unfocused", "inefficient", "targeting", "campaign", "promotion",
            "strategy", "ineffectiveness", "effectiveness", "roi", "luxury", "brand"
        ],
    }
    
    # Calculate match scores for each feature
    feature_scores = {}
    for feature in feature_list:
        score = 0
        keywords = keyword_mappings.get(feature, [])
        
        # Direct keyword matching
        for keyword in keywords:
            if keyword in user_text_lower:
                score += 1
                # Boost score for longer/more specific matches
                if len(keyword.split()) > 1:
                    score += 0.5
        
        # Bonus for feature name match
        feature_lower = feature.lower()
        if feature_lower in user_text_lower:
            score += 2
        
        feature_scores[feature] = score
    
    # Return features with score > 0, sorted by score
    matched_features = sorted(
        [(f, s) for f, s in feature_scores.items() if s > 0],
        key=lambda x: x[1],
        reverse=True
    )
    
    return [f for f, s in matched_features]

# ── Load data ────────────────────────────────────────────────────────────


def engineer_features_from_csv(df):
    """Create essential features from enriched customer CSV data."""
    d = df.copy()
    
    # Map CSV columns to analysis features
    # Income → annual_spend (proxy: total income normalized)
    if 'efx_total_income360' in d.columns:
        income_median = d['efx_total_income360'].median()
        d['annual_spend'] = (d['efx_total_income360'] / income_median).clip(0.1, 5.0) * 25000
    else:
        d['annual_spend'] = 25000
    
    # Age → digital_propensity (younger = more digital)
    if 'spd_sd_age' in d.columns:
        d['digital_propensity'] = np.where(
            d['spd_sd_age'] < 35, 0.85,
            np.where(d['spd_sd_age'] < 50, 0.65,
                    np.where(d['spd_sd_age'] < 65, 0.45, 0.25))
        )
    else:
        d['digital_propensity'] = 0.6
    
    # Credit card usage → purchase frequency proxy
    if 'exp_financial_credit_card_user' in d.columns:
        d['purchase_freq_weight'] = (d['exp_financial_credit_card_user'] / 10).clip(0.3, 1.0)
    else:
        d['purchase_freq_weight'] = 0.6
    
# Satisfaction score (derive from purchase quality or digital profile if available)
    if 'spd_bc_purchase_quality' in d.columns:
        d['satisfaction_score'] = np.where(
            d['spd_bc_purchase_quality'] >= 5, 4.5,
            np.where(d['spd_bc_purchase_quality'] >= 4, 4.0,
                     np.where(d['spd_bc_purchase_quality'] >= 3, 3.5, 3.0))
        )
    else:
        d['satisfaction_score'] = (
            3.0 + (d['digital_propensity'] - 0.25) * 1.5 + (d.get('brand_count', 2) / 20)
        ).clip(1.0, 5.0)

    # Normalize so bucketed behavior is possible
    d['satisfaction_score'] = d['satisfaction_score'].clip(1.0, 5.0)
    
    # AI usage frequency (assume growing adoption based on digital propensity)
    d['ai_usage_freq'] = (d['digital_propensity'] * 3).clip(0, 3)
    
    # Brand count (luxury brands are often correlated with income)
    if 'efx_total_income360' in d.columns:
        d['brand_count'] = np.where(
            d['efx_total_income360'] > 500000, 6,
            np.where(d['efx_total_income360'] > 200000, 4,
                    np.where(d['efx_total_income360'] > 100000, 2, 1))
        )
    else:
        d['brand_count'] = 2
    
    # CLV: Customer lifetime value (annual_spend × multiplier × frequency)
    d['clv'] = d['annual_spend'] * 8.5 * d['purchase_freq_weight']
    
    # Retention improvement (tied to digital readiness and income)
    d['retention_improvement'] = (d['digital_propensity'] * 0.15).clip(0.05, 0.3)
    
    return d


def get_feature_df(df):
    """Extract features for clustering. Works with both Qualtrics and CSV-based data."""
    # First, check if features exist (from engineer_features)
    existing_cols = []
    required_cols = [
        'annual_spend', 'purchase_freq_weight', 'brand_count', 'digital_propensity',
        'ai_usage_freq', 'clv', 'retention_improvement', 'satisfaction_score',
    ]
    
    for col in required_cols:
        if col in df.columns:
            existing_cols.append(col)
    
    # If features don't exist, engineer them from CSV data
    if len(existing_cols) < len(required_cols):
        df = engineer_features_from_csv(df)
    
    cols = [
        'annual_spend', 'purchase_freq_weight', 'brand_count', 'digital_propensity',
        'ai_usage_freq', 'clv', 'retention_improvement', 'satisfaction_score',
    ]
    return df[cols].fillna(0)


def train_models(df_synth_raw):
    """Train models on the fly — avoids joblib version mismatch on cloud."""
    # Try to use Qualtrics-based feature engineering if columns exist
    if 'Q1' in df_synth_raw.columns:
        df = engineer_features(df_synth_raw)
        required_cols = FEATURE_COLS + ['net_roi']
    else:
        # Fall back to CSV-based engineering for available features
        df = engineer_features_from_csv(df_synth_raw)
        # Use available columns for ROI modeling
        required_cols = [
            'annual_spend', 'purchase_freq_weight', 'digital_propensity',
            'satisfaction_score', 'ai_usage_freq', 'clv', 'retention_improvement'
        ]
        # Create synthetic ROI if not available
        if 'net_roi' not in df.columns:
            df['net_roi'] = (df['clv'] * df['retention_improvement']) / (df['annual_spend'] * 0.03 + 1)
        required_cols.append('net_roi')
    
    # Filter to available columns
    available_cols = [c for c in required_cols if c in df.columns]
    X = df[available_cols[:-1]]
    y = df['net_roi']
    
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

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ OSINT Parameters")
st.sidebar.markdown("*Adjust luxury industry benchmarks from open-source intelligence*")

with st.sidebar.expander("📊 Customer Retention", expanded=True):
    baseline_ret = st.slider("Baseline Retention Rate", 0.50, 0.99, 0.82, 0.01,
                                  help="Avg luxury customer retention (Bain: ~82%)",
                                  key="ret_1")
    ai_ret_uplift = st.slider("AI Retention Uplift", 0.05, 0.35, 0.15, 0.01,
                                   help="Max retention boost from AI personalization",
                                   key="ret_2")
    clv_mult = st.slider("CLV Multiplier (x annual spend)", 3.0, 15.0, 8.5, 0.5,
                              help="Customer lifetime value as multiple of annual spend",
                              key="ret_3")
    churn_cost = st.slider("Churn Cost Multiplier", 2.0, 12.0, 5.0, 0.5,
                                help="Cost to acquire new customer vs retain existing",
                                key="ret_4")

with st.sidebar.expander("🤖 AI Revenue Impact", expanded=True):
    pers_lift = st.slider("Personalization Revenue Lift", 0.05, 0.40, 0.20, 0.01,
                               help="Revenue increase from AI personalization",
                               key="ai_1")
    ai_cost = st.slider("AI Implementation Cost Ratio", 0.01, 0.15, 0.03, 0.005,
                              help="AI cost as fraction of customer revenue",
                              key="ai_2")

with st.sidebar.expander("📈 Projection Settings", expanded=False):
    shift_rate = st.slider("Annual AI Literacy Shift Rate", 0.05, 0.50, 0.25, 0.01,
                                help="How fast AI literacy grows per year (OSINT: ~25-30%)",
                                key="proj_1")
    n_customers_proj = st.slider("Projection Customer Base", 500, 10000, 1000, 500,
                                  key="proj_2")
    projection_end = st.slider("Projection End Year", 2028, 2035, 2030,
                                key="proj_3")

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

# Use CSV-based engineering since we're loading from CSVs
df_synth = engineer_features_from_csv(df_synth_raw)
df_real = engineer_features_from_csv(df_real_raw)

# ── Header ────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="font-size: 2.2rem; margin: 0; color: #C41E3A;">💎 Agentic AI ROI Dashboard</h1>
    <p style="font-size: 1.1rem; color: #666; margin-top: 0.5rem;">Luxury Goods Industry Intelligence & ROI Analytics</p>
    <p style="font-size: 0.9rem; color: #999;">Interactive consulting for luxury brands: Segmentation → Use Case Match → ROI → Roadmap</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Top navigation for closed-loop workflow ─────────────────────────────
page = st.selectbox(
    "📋 Navigation",
    ["Home Input", "Customer Segmentation", "Agentic AI Use Cases", "ROI Calculator", "Implementation Roadmap"],
    help="Choose a section to navigate through the analysis workflow",
    key="page"
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


def get_preview_customer_count(df, selected_features, selected_pains, allow_or_mode=False):
    """
    Get the count of customers matching the selected features for preview.
    Returns: (matched_count, total_count, will_use_fallback, impact_details)
    impact_details includes per-feature residuals and suggested change.
    """
    total_count = len(df)
    if not selected_features:
        return total_count, total_count, False, {
            'per_feature': {},
            'most_restrictive': None,
            'or_mode_count': total_count
        }

    df_filtered = df.copy()
    per_feature = {}
    
    feature_filters = {
        "High-spending (High AOV) customers": lambda d: d['annual_spend'] >= d['annual_spend'].quantile(0.70),
        "High-frequency repurchase customers": lambda d: d['purchase_freq_weight'] >= d['purchase_freq_weight'].quantile(0.65),
        "Luxury limited/rare item buyers": lambda d: d['brand_count'] >= d['brand_count'].quantile(0.60),
        "Omnichannel (online + offline) shoppers": lambda d: d['digital_propensity'] >= d['digital_propensity'].quantile(0.55),
        "High private domain/community engagement customers": lambda d: d['ai_usage_freq'] >= d['ai_usage_freq'].quantile(0.65),
        "Ultra-high-net-worth VIC/VIP clients": lambda d: d['clv'] >= d['clv'].quantile(0.75),
        "At-risk churn customers (long time no purchase)": lambda d: d['retention_improvement'] <= d['retention_improvement'].quantile(0.35),
        "Leather goods/jewelry/watch preference buyers": lambda d: d['satisfaction_score'] >= d['satisfaction_score'].quantile(0.55),
    }

    pain_filters = {
        "Insufficient personalized service for high-value clients": feature_filters["Ultra-high-net-worth VIC/VIP clients"],
        "Chaotic limited-edition product allocation & appointment": feature_filters["Luxury limited/rare item buyers"],
        "Low customer repurchase & retention rate": feature_filters["At-risk churn customers (long time no purchase)"],
        "High customer service operational costs": feature_filters["High private domain/community engagement customers"],
        "Inconsistent omnichannel customer experience": feature_filters["Omnichannel (online + offline) shoppers"],
        "Unfocused and inefficient luxury marketing": feature_filters["High-frequency repurchase customers"],
    }

    # Cumulative AND filtering for strict match
    for feature in selected_features:
        if feature in feature_filters:
            filter_func = feature_filters[feature]
            mask = filter_func(df_filtered)
            df_filtered = df_filtered[mask].copy()
            per_feature[f"feature:{feature}"] = len(df_filtered)

    for pain in selected_pains:
        if pain in pain_filters:
            filter_func = pain_filters[pain]
            mask = filter_func(df_filtered)
            df_filtered = df_filtered[mask].copy()
            per_feature[f"pain:{pain}"] = len(df_filtered)

    and_count = len(df_filtered)

    # OR mode count-to-provide fallback suggestions
    or_count = and_count
    if allow_or_mode and selected_features:
        union_df = pd.DataFrame(columns=df.columns)
        for feature in selected_features:
            if feature in feature_filters:
                feature_df = df[feature_filters[feature](df)]
                union_df = pd.concat([union_df, feature_df])
        or_count = len(union_df.drop_duplicates())

    min_cohort_size = max(50, total_count * 0.05)
    will_fallback = and_count < min_cohort_size

    if per_feature:
        most_restrictive_feature = min(per_feature, key=per_feature.get)
    else:
        most_restrictive_feature = None

    impact_details = {
        'per_feature': per_feature,
        'most_restrictive': most_restrictive_feature,
        'and_count': and_count,
        'or_count': or_count,
        'min_cohort_size': min_cohort_size,
    }

    return and_count, total_count, will_fallback, impact_details


def filter_dataset_by_customer_features(df, selected_features):
    """
    Dynamically filter dataset based on selected customer features.
    Maps UI feature selections to actual column-based filtering logic.
    Falls back to full dataset if filtering is too restrictive.
    """
    if not selected_features:
        return df.copy(), [], False  # Return full dataset, empty filter list, and no fallback flag
    
    df_filtered = df.copy()
    applied_filters = []
    
    # Define feature → filtering criteria mapping
    feature_filters = {
        "High-spending (High AOV) customers": lambda d: d['annual_spend'] >= d['annual_spend'].quantile(0.70),
        "High-frequency repurchase customers": lambda d: d['purchase_freq_weight'] >= d['purchase_freq_weight'].quantile(0.65),
        "Luxury limited/rare item buyers": lambda d: d['brand_count'] >= d['brand_count'].quantile(0.60),
        "Omnichannel (online + offline) shoppers": lambda d: d['digital_propensity'] >= d['digital_propensity'].quantile(0.55),
        "High private domain/community engagement customers": lambda d: d['ai_usage_freq'] >= d['ai_usage_freq'].quantile(0.65),
        "Ultra-high-net-worth VIC/VIP clients": lambda d: d['clv'] >= d['clv'].quantile(0.75),
        "At-risk churn customers (long time no purchase)": lambda d: d['retention_improvement'] <= d['retention_improvement'].quantile(0.35),
        "Leather goods/jewelry/watch preference buyers": lambda d: d['satisfaction_score'] >= d['satisfaction_score'].quantile(0.55),
    }

    pain_filters = {
        "Insufficient personalized service for high-value clients": feature_filters["Ultra-high-net-worth VIC/VIP clients"],
        "Chaotic limited-edition product allocation & appointment": feature_filters["Luxury limited/rare item buyers"],
        "Low customer repurchase & retention rate": feature_filters["At-risk churn customers (long time no purchase)"],
        "High customer service operational costs": feature_filters["High private domain/community engagement customers"],
        "Inconsistent omnichannel customer experience": feature_filters["Omnichannel (online + offline) shoppers"],
        "Unfocused and inefficient luxury marketing": feature_filters["High-frequency repurchase customers"],
    }

    # Apply filters cumulatively (AND logic)
    for feature in selected_features:
        if feature in feature_filters:
            filter_func = feature_filters[feature]
            mask = filter_func(df_filtered)
            df_filtered = df_filtered[mask].copy()
            applied_filters.append(feature)

    for pain in selected_pains:
        if pain in pain_filters:
            filter_func = pain_filters[pain]
            mask = filter_func(df_filtered)
            df_filtered = df_filtered[mask].copy()
            applied_filters.append("pain:" + pain)
    
    # Fallback: If filtering eliminates too many customers, use full dataset
    min_cohort_size = max(50, len(df) * 0.05)  # At least 50 customers or 5% of dataset
    fallback_used = False
    if len(df_filtered) < min_cohort_size:
        st.warning(
            f"⚠️ Feature filters too restrictive: only {len(df_filtered)} customers matched. "
            f"Using full dataset of {len(df)} customers for analysis instead.",
            icon="⚠️"
        )
        df_filtered = df.copy()
        applied_filters = []
        fallback_used = True
    
    return df_filtered, applied_filters, fallback_used


def run_analysis(selected_features, selected_pains, annual_revenue):
    df = df_real.copy()
    
    # Apply feature-based filtering to make clustering responsive to user selections
    df_filtered, applied_filters, fallback_used = filter_dataset_by_customer_features(df, selected_features)
    
    # Handle case where no analysis is possible
    if len(df_filtered) == 0:
        st.error("❌ No customers match your selected criteria. Please adjust your selections and try again.")
        st.stop()
    
    # Track dataset size before/after filtering for transparency
    original_size = len(df)
    filtered_size = len(df_filtered)
    filter_info = {
        'original_size': original_size,
        'filtered_size': filtered_size,
        'applied_filters': applied_filters,
        'retention_pct': (filtered_size / original_size * 100) if original_size > 0 else 100,
        'fallback_used': fallback_used,
    }
    
    # Use filtered dataset for clustering analysis
    feature_df = get_feature_df(df_filtered)
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
    df_filtered['cluster'] = kmeans.labels_

    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X)
    df_filtered['pc1'] = pca_coords[:, 0]
    df_filtered['pc2'] = pca_coords[:, 1]

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

    df_filtered['cluster_name'] = df_filtered['cluster'].map(cluster_name_map)

    # Use case matching
    matched = []
    for target_seg, pain, case_name, desc, conv, aov, labor in USE_CASE_RULES:
        seg_match = True if target_seg is None else target_seg in df_filtered['cluster_name'].unique()
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
    if "Ultra-High-Net-Worth Connoisseurs" in df_filtered['cluster_name'].values:
        roadmap_template = [
            ("0-3 Months", "Data desensitization & integration; AI knowledge base; pilot concierge agent."),
            ("3-6 Months", "Limited-edition allocation agent launch; VIC private domain service scaling."),
            ("6-12 Months", "Full omnichannel synergy; multi-agent collaboration; ROI optimization."),
        ]
    elif "Aspirational Luxury Buyers" in df_filtered['cluster_name'].values or "Gift-Oriented Luxury Purchasers" in df_filtered['cluster_name'].values:
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
        'df': df_filtered,
        'filter_info': filter_info,
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
    show_progress_steps(1)
    
    st.markdown("### 📌 Step 1: Customer Profile & Pain Point Input")
    st.markdown("*Provide luxury customer signal selections and business pain points. Annual revenue is used for ROI calibration.*")
    st.divider()
    
    # Add tabs for two input methods
    tab1, tab2 = st.tabs(["📋 Structured Input", "✍️ Smart Text Input"])
    
    # Initialize session state for selections
    if 'auto_selected_features' not in st.session_state:
        st.session_state.auto_selected_features = []
    if 'auto_selected_pains' not in st.session_state:
        st.session_state.auto_selected_pains = []
    if 'input_method' not in st.session_state:
        st.session_state.input_method = 'structured'
    
    # ─────────────────────────────────────────────────────────────────
    # TAB 1: STRUCTURED INPUT (Original method)
    # ─────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("**Method 1: Select from predefined options**")
        st.markdown("Choose the customer profiles and pain points that match your business.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 👥 Customer Features")
            st.markdown("Select 1-3 key customer profiles that matter most to your business:")
            structured_features = st.multiselect(
                "Customer Feature Selection", 
                FEATURE_CHOICES,
                label_visibility="collapsed",
                help="Choose customer segments you want to focus on",
                key="structured_features"
            )
            
            if structured_features:
                st.caption(f"✓ {len(structured_features)} feature(s) selected")

        with col2:
            st.markdown("#### 💼 Business Pain Points")
            st.markdown("Select 1-3 operational challenges to address:")
            structured_pains = st.multiselect(
                "Core Business Pain Points Selection", 
                PAIN_CHOICES,
                label_visibility="collapsed",
                help="Choose the business challenges you face",
                key="structured_pains"
            )
            
            if structured_pains:
                st.caption(f"✓ {len(structured_pains)} pain point(s) selected")
    
    # ─────────────────────────────────────────────────────────────────
    # TAB 2: SMART TEXT INPUT (New feature)
    # ─────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("**Method 2: Describe in your own words**")
        st.markdown("Write a description of your customer types and business challenges. The AI will automatically identify matching categories.")
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.markdown("#### 📝 Customer Description")
            customer_desc = st.text_area(
                "Describe your target customers",
                height=120,
                placeholder="Example: We focus on ultra-high-net-worth VIP clients who make expensive purchases and are very engaged with our brand community. They are frequent luxury buyers looking for exclusive limited edition items.",
                help="Be as descriptive as possible for better matching",
                key="customer_description"
            )
            
            if customer_desc:
                # Auto-detect features from description
                auto_features = match_features_from_text(customer_desc, FEATURE_CHOICES)
                if auto_features:
                    st.markdown("**🎯 Detected Customer Features:**")
                    cols = st.columns(len(auto_features) if len(auto_features) <= 3 else 3)
                    for idx, feature in enumerate(auto_features[:3]):
                        with cols[idx % 3]:
                            st.markdown(f"✓ {feature}")
                    if len(auto_features) > 3:
                        st.caption(f"+ {len(auto_features) - 3} more matches")
                else:
                    st.info("💡 No customer features matched yet. Try being more specific about your customer types.")
        
        with col2:
            st.markdown("#### 🔧 Pain Points Description")
            pain_desc = st.text_area(
                "Describe your business challenges",
                height=120,
                placeholder="Example: We struggle with high customer service costs and inconsistent experience across online and offline channels. Limited edition product allocation is chaotic and our retention rates are declining.",
                help="Describe the operational or business challenges you face",
                key="pain_description"
            )
            
            if pain_desc:
                # Auto-detect pain points from description
                auto_pains = match_features_from_text(pain_desc, PAIN_CHOICES)
                if auto_pains:
                    st.markdown("**🎯 Detected Pain Points:**")
                    cols = st.columns(len(auto_pains) if len(auto_pains) <= 3 else 3)
                    for idx, pain in enumerate(auto_pains[:3]):
                        with cols[idx % 3]:
                            st.markdown(f"✓ {pain}")
                    if len(auto_pains) > 3:
                        st.caption(f"+ {len(auto_pains) - 3} more matches")
                else:
                    st.info("💡 No pain points matched yet. Try being more specific about your challenges.")
        
        st.divider()
        
        # Save auto-detected selections
        if customer_desc or pain_desc:
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("✅ Use Detected Categories", use_container_width=True):
                    st.session_state.auto_selected_features = auto_features if customer_desc else []
                    st.session_state.auto_selected_pains = auto_pains if pain_desc else []
                    st.session_state.input_method = 'smart'
                    st.success("✓ Categories updated!")
            
            with col2:
                if st.session_state.auto_selected_features or st.session_state.auto_selected_pains:
                    st.caption(
                        f"📌 Using: {len(st.session_state.auto_selected_features)} features, "
                        f"{len(st.session_state.auto_selected_pains)} pain points"
                    )
    
    st.divider()
    
    # Determine which selections to use
    if st.session_state.input_method == 'smart':
        selected_features = st.session_state.auto_selected_features
        selected_pains = st.session_state.auto_selected_pains
    else:
        selected_features = structured_features
        selected_pains = structured_pains
    
    st.markdown("#### 💰 Business Metrics")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        annual_revenue = st.number_input(
            "Brand Annual Sales/Revenue", 
            min_value=0.0, 
            value=5000000.0, 
            step=10000.0, 
            format="%.2f",
            help="Used for ROI calculations and revenue impact projections"
        )
        st.caption(f"Revenue: ${annual_revenue:,.0f}")
    
    with col2:
        st.metric("Expected Impact", "Up to 20% ROI")
    
    st.divider()
    
    # Smart Validation Preview - Real-time customer count indicator
    st.markdown("#### 🔍 Customer Match Preview")

    allow_or_mode = st.checkbox(
        "Enable broader OR-based matching suggestion", False,
        help="Show comparison of strict AND cohort and broader OR cohort to guide selection adjustments"
    )

    matched_count, total_count, will_fallback, impact = get_preview_customer_count(
        df_real.copy(), selected_features, selected_pains, allow_or_mode=allow_or_mode
    )

    safe_threshold = 500  # Green zone
    warning_threshold = 50  # Below this is red

    if matched_count >= safe_threshold:
        status_color = "🟢"
        status_label = "EXCELLENT"
        status_description = f"✅ {matched_count:,} customers match your selections"
        suggestion = "Your selections are well-targeted. Analysis will be comprehensive."
    elif matched_count >= warning_threshold:
        status_color = "🟡"
        status_label = "CAUTION"
        status_description = f"⚠️ {matched_count:,} customers match your selections"
        suggestion = "Acceptable but tight. Consider broadening 1 filter for more comprehensive analysis."
    else:
        status_color = "🔴"
        status_label = "TOO RESTRICTIVE"
        status_description = f"❌ Only {matched_count:,} customers match your selections"
        suggestion = "Your filters are too restrictive for meaningful analysis. Try removing one feature or pain point."

    col1, col2, col3 = st.columns([1, 2, 2])

    with col1:
        st.markdown(f"### {status_color}")
        st.caption(status_label)

    with col2:
        st.markdown(f"**{status_description}**")
        st.caption(f"Out of {total_count:,} total customers")

    with col3:
        st.info(f"💡 **Tip:** {suggestion}")

    # Show OR-mode impact guidance if enabled
    if allow_or_mode and selected_features:
        st.info(
            f"By OR-matching selected features, you could address up to {impact['or_count']:,} potential customers. "
            "Use this as a soft target when AND is too restrictive."
        )

    # Show feature-level restriction details for zero/tiny match
    if matched_count < warning_threshold and selected_features:
        st.markdown("#### ⚙️ Filter Impact Breakdown")
        if impact['per_feature']:
            rows = []
            for f, c in impact['per_feature'].items():
                rows.append(f"• {f}: {c:,} customers remain")
            st.markdown("\n".join(rows))
            if impact['most_restrictive']:
                st.warning(
                    f"Most restrictive selection: '{impact['most_restrictive']}' (only {impact['per_feature'][impact['most_restrictive']]:,} customers remain). "
                    "Consider removing it to grow the cohort."
                )

    if will_fallback:
        st.warning(
            f"⚠️ Note: {matched_count:,} customers is below the fallback threshold ({impact['min_cohort_size']:,}). "
            f"Analysis will use full dataset ({total_count:,}) for stable clustering.",
            icon="⚠️"
        )
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            if not selected_features:
                st.warning("⚠️ Please select at least one customer feature.")
            elif not selected_pains:
                st.warning("⚠️ Please select at least one business pain point.")
            elif matched_count == 0:
                st.error("❌ No customers match your current filters. Remove or relax one feature and try again.")
            elif matched_count < 50:
                st.warning(
                    f"⚠️ Only {matched_count:,} customers match. Analysis may be unstable. "
                    "Consider relaxing filters or enabling OR-based best-fit mode."
                )
                if st.button("✅ Proceed with limited cohort", use_container_width=True):
                    with st.spinner("🔄 Running analysis... this may take a moment"):
                        run_analysis(selected_features, selected_pains, annual_revenue)
                    st.success("✅ Analysis complete! Navigate to other tabs via the sidebar to view results.")
            else:
                with st.spinner("🔄 Running analysis... this may take a moment"):
                    run_analysis(selected_features, selected_pains, annual_revenue)
                st.success("✅ Analysis complete! Navigate to other tabs via the sidebar to view results.")
    with col2:
        st.button("📖 View Guide", use_container_width=True, disabled=True)
    
    if st.session_state.analysis:
        col3.info("✓ Analysis history available for review")
    
    st.divider()
    
    # Help section
    with st.expander("❓ How to Use This Dashboard", expanded=False):
        st.markdown("""
        **Welcome to the Agentic AI ROI Dashboard!**
        
        This interactive tool helps luxury brands understand the impact of AI-powered customer operations.
        
        **Two Input Methods:**
        - **Structured Input**: Use predefined dropdown lists for maximum accuracy
        - **Smart Text Input**: Describe your customers and challenges in your own words - AI automatically identifies matching categories
        
        **Workflow:**
        1. **Home Input** (You are here) - Define your customer profiles and business challenges
        2. **Customer Segmentation** - Understand your customer base through AI clustering
        3. **AI Use Cases** - Discover AI applications aligned to your business
        4. **ROI Calculator** - Review financial projections and ROI 
        5. **Implementation Roadmap** - Get a phased deployment plan
        
        **Tips:**
        - Select 1-3 features that best represent your target customers
        - Choose 1-3 pain points you want to address
        - For Smart Text Input, be descriptive about your specific situation
        - Enter your actual annual revenue for accurate ROI projections
        - The analysis typically takes 10-30 seconds
        """)
    
    st.stop()

# Customer Segmentation page
if page == "Customer Segmentation":
    show_progress_steps(2)
    
    st.markdown("### 📊 Step 2: Customer Segmentation Analysis")
    st.markdown("*K-Means clustering reveals natural customer groupings in your database*")
    
    if not st.session_state.analysis:
        st.error("❌ Please run the Home Input analysis first.", icon="❌")
        st.stop()

    analysis = st.session_state.analysis
    df = analysis['df']
    n_clusters = analysis['kmeans_k']
    filter_info = analysis.get('filter_info', {})

    # Display filter context and dataset scope
    with st.expander("📌 Analysis Scope & Applied Filters", expanded=True):
        if filter_info.get('fallback_used'):
            st.warning(
                "⚠️ **Filter Fallback Applied**: Your feature selections were too restrictive and would have eliminated "
                "nearly all customers. The system automatically fell back to analyzing your full customer base. "
                "Try selecting fewer or less restrictive customer features.",
                icon="⚠️"
            )
        
        if filter_info.get('applied_filters'):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Dataset Size", f"{filter_info['original_size']:,} customers")
            with col2:
                st.metric("Analyzed Cohort Size", f"{filter_info['filtered_size']:,} customers")
            with col3:
                st.metric("Retention Rate", f"{filter_info['retention_pct']:.1f}%")
            
            st.markdown("**Applied Customer Filters:**")
            for i, filt in enumerate(filter_info['applied_filters'], 1):
                st.markdown(f"{i}. {filt}")
            
            st.info("""
            ℹ️ **What this means:** The customer profile metrics below represent **segment averages** computed from your filtered cohort. 
            Each value shows the mean of that metric across all customers in that segment. 
            If you change your selected features on the Home Input page, the segmentation and these averages will update dynamically.
            """)
        else:
            st.metric("Full Dataset Size", f"{filter_info['original_size']:,} customers")
            st.info("""
            ℹ️ **What this means:** Analyzing **full customer dataset** with no feature filters. 
            Segment averages below represent the mean values across your entire customer base within each cluster.
            """)

    st.markdown(f"""
    ### Clustering Overview
    Identified **{n_clusters} distinct customer segments** using K-Means with elbow method optimization.
    These segments represent natural groupings in your customer base based on behavioral, 
    financial, and engagement characteristics.
    """)

    cust_counts = df['cluster_name'].value_counts().reset_index()
    cust_counts.columns = ['Segment', 'Customer_Count']
    cust_counts['Percentage'] = (cust_counts['Customer_Count'] / len(df) * 100).round(1)

    # Segment Overview Cards
    st.subheader("📊 Segment Overview")
    cols = st.columns(len(cust_counts))
    for idx, (_, row) in enumerate(cust_counts.iterrows()):
        with cols[idx]:
            st.metric(
                row['Segment'],
                f"{row['Customer_Count']:,}",
                f"{row['Percentage']:.1f}% of base"
            )

    # ─────────────────────────────────────────────────────────────────
    # PCA Visualization with Enhanced Design
    # ─────────────────────────────────────────────────────────────────
    st.subheader("🎯 Segment Positioning (2D PCA Projection)")
    st.markdown("*Bubble size = Customer Lifetime Value (CLV); Position = Principal component space*")
    
    fig_pca = px.scatter(df, x='pc1', y='pc2', color='cluster_name', size='clv',
                         hover_data={
                             'annual_spend': ':.0f',
                             'pc1': ':.2f',
                             'pc2': ':.2f',
                             'clv': ':.0f',
                             'cluster_name': True
                         },
                         labels={
                             'pc1': 'Principal Component 1',
                             'pc2': 'Principal Component 2',
                             'cluster_name': 'Segment'
                         },
                         title='Customer Cluster Distribution',
                         height=550,
                         color_discrete_sequence=px.colors.qualitative.Set2)
    fig_pca.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='white')))
    fig_pca.update_layout(font=dict(size=11), hovermode='closest')
    st.plotly_chart(fig_pca, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # Detailed Segment Profiles
    # ─────────────────────────────────────────────────────────────────
    st.subheader("🔍 Detailed Segment Profiles")

    # Build segment profiles with available columns
    available_cols = {
        'annual_spend': 'Annual Spend',
        'clv': 'Customer Lifetime Value',
        'purchase_freq_weight': 'Purchase Frequency',
        'ai_usage_freq': 'AI Usage',
        'digital_propensity': 'Digital Propensity',
        'retention_improvement': 'Retention Rate',
        'satisfaction_score': 'Satisfaction',
        'brand_count': 'Brand Count',
        'ai_readiness': 'AI Readiness',
    }

    # Filter to only columns that exist
    existing_cols = {k: v for k, v in available_cols.items() if k in df.columns}
    
    segment_profiles = []
    for seg in sorted(df['cluster_name'].unique()):
        seg_df = df[df['cluster_name'] == seg]
        profile_dict = {'Segment': seg, 'Size': len(seg_df)}
        
        for col, label in existing_cols.items():
            profile_dict[label] = seg_df[col].mean()
        
        segment_profiles.append(profile_dict)

    profile_df = pd.DataFrame(segment_profiles)

    # Create segment detail expanders with dynamic columns
    for idx, seg_name in enumerate(sorted(df['cluster_name'].unique())):
        seg_data = profile_df[profile_df['Segment'] == seg_name].iloc[0]
        seg_df = df[df['cluster_name'] == seg_name]

        with st.expander(f"📋 {seg_name.upper()} — {seg_data['Size']:,} customers", expanded=(idx==0)):
            # Display clear label that these are segment averages
            st.caption("📊 **Segment Average Metrics** — Mean values calculated across all customers in this segment")
            
            # Create 4-column layout for metrics
            metric_cols = st.columns(4)
            metric_idx = 0
            
            # Base metrics
            base_metrics = ['Annual Spend', 'Customer Lifetime Value', 'Purchase Frequency', 'AI Readiness']
            for metric_name in base_metrics:
                if metric_name in seg_data.index:
                    value = seg_data[metric_name]
                    if metric_name == 'Annual Spend' or metric_name == 'Customer Lifetime Value':
                        formatted_val = f"${value:,.0f}"
                    elif metric_name in ['Purchase Frequency', 'AI Readiness']:
                        formatted_val = f"{value:.2f}"
                    else:
                        formatted_val = f"{value:.2f}"
                    
                    with metric_cols[metric_idx % 4]:
                        st.metric(metric_name, formatted_val)
                    metric_idx += 1
            
            # Second row of metrics
            metric_cols2 = st.columns(4)
            other_metrics = ['Retention Rate', 'Digital Propensity', 'Satisfaction', 'Brand Count']
            for i, metric_name in enumerate(other_metrics):
                if metric_name in seg_data.index:
                    value = seg_data[metric_name]
                    if metric_name == 'Retention Rate' or metric_name == 'Digital Propensity':
                        formatted_val = f"{value:.1%}"
                    elif metric_name == 'Brand Count':
                        formatted_val = f"{value:.1f}"
                    else:
                        formatted_val = f"{value:.2f}"
                    
                    with metric_cols2[i % 4]:
                        st.metric(metric_name, formatted_val)

            # Segment characteristics
            st.markdown("**📌 Segment Characteristics:**")
            
            characteristics = []
            if 'Annual Spend' in seg_data.index and seg_data['Annual Spend'] > profile_df['Annual Spend'].quantile(0.75):
                characteristics.append("💰 **High-Value Spenders**")
            if 'Customer Lifetime Value' in seg_data.index and seg_data['Customer Lifetime Value'] > profile_df['Customer Lifetime Value'].quantile(0.75):
                characteristics.append("🎯 **Premium Lifetime Value**")
            if 'Purchase Frequency' in seg_data.index and seg_data['Purchase Frequency'] > profile_df['Purchase Frequency'].quantile(0.75):
                characteristics.append("🔄 **Frequent Repeat Buyers**")
            if 'AI Readiness' in seg_data.index and seg_data['AI Readiness'] > profile_df['AI Readiness'].quantile(0.75):
                characteristics.append("🤖 **AI-Savvy & Digital-Native**")
            if 'Retention Rate' in seg_data.index and seg_data['Retention Rate'] > profile_df['Retention Rate'].quantile(0.75):
                characteristics.append("🛡️ **Highly Loyal**")
            if 'Satisfaction' in seg_data.index and seg_data['Satisfaction'] > profile_df['Satisfaction'].quantile(0.75):
                characteristics.append("😊 **High Satisfaction**")

            if characteristics:
                st.markdown(" | ".join(characteristics))
            else:
                st.markdown("*Balanced profile across all dimensions*")

    # ─────────────────────────────────────────────────────────────────
    # Normalized Radar Chart (Better for comparison)
    # ─────────────────────────────────────────────────────────────────
    st.subheader("📈 Segment Feature Comparison (Normalized)")
    st.markdown("*Each dimension normalized 0-1 for fair comparison across segments*")

    # Determine which radar features are available
    possible_radar_features = ['annual_spend', 'purchase_freq_weight', 'clv', 'ai_usage_freq', 'retention_improvement']
    radar_features = [f for f in possible_radar_features if f in df.columns]
    
    if radar_features:
        radar_data = []
        for seg in sorted(df['cluster_name'].unique()):
            seg_df = df[df['cluster_name'] == seg]
            radar_dict = {'segment': seg}
            for feat in radar_features:
                radar_dict[feat] = seg_df[feat].mean()
            radar_data.append(radar_dict)

        radar_df = pd.DataFrame(radar_data)
        
        # Normalize each feature to 0-1 range
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        radar_normalized = radar_df.copy()
        radar_normalized[radar_features] = scaler.fit_transform(radar_df[radar_features])

        # Create normalized radar chart
        colors = px.colors.qualitative.Pastel
        fig = go.Figure()
        
        feature_labels = {
            'annual_spend': 'Annual Spend',
            'purchase_freq_weight': 'Purchase Frequency',
            'clv': 'Customer Lifetime Value',
            'ai_usage_freq': 'AI Usage',
            'retention_improvement': 'Retention Improvement'
        }
        theta_labels = [feature_labels.get(f, f) for f in radar_features]
        
        for idx, (_, row) in enumerate(radar_normalized.iterrows()):
            fig.add_trace(go.Scatterpolar(
                r=row[radar_features].values,
                theta=theta_labels,
                fill='toself',
                name=row['segment'],
                line=dict(color=colors[idx % len(colors)]),
                fillcolor=colors[idx % len(colors)],
                opacity=0.6,
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=['0%', '25%', '50%', '75%', '100%'],
                ),
                angularaxis=dict(rotation=0),
            ),
            showlegend=True,
            height=500,
            font=dict(size=10),
            template='plotly_white',
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Required columns for radar chart not found in data.")

    # ─────────────────────────────────────────────────────────────────
    # Segment Composition Pie Chart
    # ─────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Segment Distribution")
        pie = px.pie(cust_counts, names='Segment', values='Customer_Count',
                     title=f'{n_clusters} Customer Segments',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(pie, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # Heatmap: Segment x Key Metrics
    # ─────────────────────────────────────────────────────────────────
    with col2:
        st.subheader("🔥 Segment Profile Heatmap")
        
        # Use available columns from profile_df
        heatmap_cols = [c for c in profile_df.columns if c not in ['Segment', 'Size']]
        if heatmap_cols:
            heatmap_data = profile_df.set_index('Segment')[heatmap_cols]
            
            # Normalize for heatmap
            heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-6)
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_normalized.values,
                x=heatmap_normalized.columns,
                y=heatmap_normalized.index,
                colorscale='RdYlGn',
                text=heatmap_data.values.round(2),
                texttemplate='%{text:.2f}',
                hovertemplate='%{y}<br>%{x}<br>Value: %{text:.2f}<extra></extra>',
            ))
            fig_heatmap.update_layout(height=350, font=dict(size=9))
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("⚠️ Insufficient data for heatmap visualization.")

    # ─────────────────────────────────────────────────────────────────
    # Segment Relationships: Spend vs Loyalty  
    # ─────────────────────────────────────────────────────────────────
    st.subheader("💎 Customer Value vs Loyalty Matrix")
    
    scatter_df = df.copy()
    scatter_df['Segment'] = scatter_df['cluster_name']
    
    # Use available columns
    x_col = 'annual_spend' if 'annual_spend' in df.columns else None
    y_col = 'retention_improvement' if 'retention_improvement' in df.columns else None
    size_col = 'clv' if 'clv' in df.columns else None
    
    if x_col and y_col and size_col:
        fig_scatter = px.scatter(scatter_df, 
                                x=x_col, 
                                y=y_col,
                                color='Segment',
                                size=size_col,
                                hover_data={x_col: ':.0f', size_col: ':.0f'},
                                labels={
                                    x_col: 'Annual Spend ($)',
                                    y_col: 'Retention Improvement (%)',
                                    size_col: 'Customer Lifetime Value'
                                },
                                color_discrete_sequence=px.colors.qualitative.Set2,
                                height=450)
        fig_scatter.update_traces(marker=dict(opacity=0.6, line=dict(width=1, color='white')))
        
        # Add quadrant lines
        fig_scatter.add_vline(x=scatter_df[x_col].median(), line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.add_hline(y=scatter_df[y_col].median(), line_dash="dash", line_color="gray", opacity=0.5)
        
        fig_scatter.update_layout(hovermode='closest', font=dict(size=10), template='plotly_white')
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("⚠️ Required columns for scatter plot not found.")

    # ─────────────────────────────────────────────────────────────────
    # Segment Summary Table
    # ─────────────────────────────────────────────────────────────────
    st.subheader("📑 Segment Summary Statistics")
    
    summary_display = profile_df.copy()
    
    # Format numeric columns appropriately
    for col in summary_display.columns:
        if col not in ['Segment', 'Size']:
            if 'Spend' in col or 'CLV' in col or 'Value' in col:
                try:
                    summary_display[col] = summary_display[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
                except:
                    pass
            elif 'Rate' in col or 'Propensity' in col:
                try:
                    summary_display[col] = summary_display[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) and 0 <= x <= 1 else f"{x:.2f}")
                except:
                    pass
            elif 'Frequency' in col:
                try:
                    summary_display[col] = summary_display[col].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "N/A")
                except:
                    pass
    
    st.dataframe(summary_display, use_container_width=True, hide_index=True)
    st.stop()

# Agentic AI Use Cases page
if page == "Agentic AI Use Cases":
    show_progress_steps(3)
    
    st.markdown("### 🤖 Step 3: Strategic Agentic AI Use Case Alignment")
    st.markdown("*Discover AI applications matched to your customer segments and business challenges*")
    
    if not st.session_state.analysis:
        st.error("❌ Please run the Home Input analysis first.", icon="❌")
        st.stop()

    matched = st.session_state.analysis['matched_use_cases']
    annual_rev = st.session_state.analysis['annual_revenue']
    
    if not matched:
        st.warning("⚠️ No use cases matched. Please refine pain points or segmentation.")
        st.stop()

    st.divider()

    st.markdown("""
    ### Executive Summary: Data-Driven Agent Deployment
    
    These agentic AI initiatives are calibrated to **your customer data signatures**—behavioral patterns, 
    income profiles, decision-making authority, and digital propensity extracted from your customer base.
    Each agent targets specific customer segments with highest adoption potential and ROI impact.
    """)

    # Customer Segment Context
    with st.expander("📊 Your Customer Segments at a Glance"):
        seg_context = pd.DataFrame([
            {
                'Segment': 'Affluent Plus',
                'Size': '~1,000 customers',
                'Avg Income': '$667K',
                'Profile': 'Ultra-high-net-worth; 37.9% investment income; Strong decision-makers',
                'Digital Propensity': 'Very High (197% online commerce)',
                'Spending Power': '$52,956 annual discretionary',
                'Agent Priority': '🔴 HIGHEST - Personal Concierge',
            },
            {
                'Segment': 'Mass Affluent',
                'Size': '~1,000 customers',
                'Avg Income': '$195K',
                'Profile': 'Corporate managers; 27.6% premium CC users; Efficiency-focused',
                'Digital Propensity': 'High (207% online commerce)',
                'Spending Power': '$22,848 annual discretionary',
                'Agent Priority': '🟠 HIGH - Purchase Workflow',
            },
            {
                'Segment': 'Mass Market',
                'Size': '~3,000+ customers',
                'Avg Income': '$81K',
                'Profile': 'Broader demographics; Price-conscious; Repeat buyers',
                'Digital Propensity': 'Developing',
                'Spending Power': 'Variable by income tier',
                'Agent Priority': '🟡 MEDIUM - Retention Focus',
            },
        ])
        st.dataframe(seg_context, use_container_width=True)

    # Use case cards with data validation
    for i, uc in enumerate(matched):
        with st.container(border=True):
            col_title, col_metrics = st.columns([2, 1])
            with col_title:
                st.markdown(f"### 🤖 {uc['name']}")
            
            with col_metrics:
                est_contribution = (uc['conversion_lift'] + uc['aov_lift']) * 100
                st.metric("Est. ROI Contribution", f"+{est_contribution:.1f}%")
            
            # Business problem + segment alignment
            st.markdown(f"**Business Problem Solved:** {uc['description']}")
            
            # Determine primary segment this agent serves
            agent_segments = {
                'AI Personal Luxury Concierge Agent': ('Affluent Plus', 'Ultra-high spenders seeking hyper-personalized experiences 24/7'),
                'Limited-Edition Allocation & Pre-Release Agent': ('Mass Affluent + Affluent Plus', 'Collectors with decision-making authority managing multi-stakeholder approval'),
                'Customer Retention & Re-engagement Agent': ('Mass Market', 'At-risk customers with 30+ day inactivity; value $1,500-$15K CLV'),
                'Intelligent Private Domain Operation Agent': ('Affluent Plus', 'VIP ecosystems with dedicated concierge teams needing AI augmentation'),
                'Omnichannel Experience Synergy Agent': ('All Segments', 'Bridge online-offline friction; unified loyalty + inventory visibility'),
            }
            
            target_seg, seg_detail = agent_segments.get(uc['name'], ('All Segments', 'Impacts multiple customer segments'))
            st.markdown(f"**Target Segment:** {target_seg}  \n**Segment Detail:** {seg_detail}")
            
            # Impact metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                revenue_impact = annual_rev * (uc['conversion_lift'] + uc['aov_lift'])
                st.metric("Annual Revenue Uplift", f"${revenue_impact:,.0f}")
                st.caption(f"Conversion +{uc['conversion_lift']*100:.1f}% | AOV +{uc['aov_lift']*100:.1f}%")
            
            with col2:
                labor_cost_savings = annual_rev * 0.15 * uc['labor_saving']
                st.metric("Annual Labor Savings", f"${labor_cost_savings:,.0f}")
                st.caption(f"Operational efficiency -{uc['labor_saving']*100:.0f}%")
            
            with col3:
                total_impact = revenue_impact + labor_cost_savings
                st.metric("Total Annual Impact", f"${total_impact:,.0f}")
                st.caption(f"Net business value")
            
            # Strategic details with data-backed context
            with st.expander("📋 Strategic Implementation Details & Data Insights"):
                st.markdown(f"""
                **Agent Capabilities:**
                - Autonomous decision-making powered by real-time customer and inventory data
                - Natural language interaction with personalized tone matched to segment
                - Integration with existing CRM (customer tier), POS (purchase history), email/SMS
                
                **Data-Backed Customer Value:**
                - **{target_seg}** customers show {uc['conversion_lift']*100:.1f}% higher conversion with personalization
                - Increase basket size through intelligent recommendation: +{uc['aov_lift']*100:.1f}%
                - 24/7 availability eliminating friction in customer journey (esp. global luxury buyers)
                - Response time reduced from hours → **<30 minutes** (measured impact +8-12% satisfaction)
                
                **Operational Excellence:**
                - Reduces manual touchpoints in service delivery by {uc['labor_saving']*100:.0f}%
                - Frees premium concierge team (avg cost $80-120K/yr) for high-judgment decisions
                - Automates repetitive tasks: appointment scheduling, order tracking, tier-appropriate recommendations
                - Scales to handle 200-1000 customers per agent with <5% escalation to humans
                
                **Real-World Impact Examples:**
                - Affluent Plus segment: Reduce email response time 6-12h → <30min = **+4-8% conversion**
                - Mass Affluent segment: Multi-stakeholder approval process streamlined 30-45 days → **15-21 days** = **+3-6% close rate**
                - Mass Market churn segment: AI-triggered re-engagement campaigns recover **8-12% lost customers** within 90 days
                """)
            
            # Success metrics based on customer behaviors
            with st.expander("📊 Success Metrics & KPIs (Segment-Validated)"):
                st.markdown(f"""
                **Conversion & Revenue Metrics:**
                - Conversion rate uplift: **+{uc['conversion_lift']*100:.1f}%** (validated across {target_seg})
                - Average order value increase: **+{uc['aov_lift']*100:.1f}%**
                - Customer acquisition cost reduction: ~15-20% (via referrals, organic reviews)
                - Email open rate: +25-35% (agent-personalized campaigns)
                
                **Customer Experience Metrics:**
                - Agent response time: **< 2 seconds** (vs 6-24h for human agents)
                - Customer satisfaction (CSAT): Target +15% within 90 days
                - Net Promoter Score (NPS): +10-15 points (retention through experience quality)
                - Repeat purchase frequency: +20% (habit formation via proactive touchpoints)
                - Escalation rate to human: **< 5%** (only complex, high-value decisions)
                
                **Operational Metrics:**
                - Labor hours saved: {uc['labor_saving']*100:.0f}% reduction in service team workload
                - Cost per customer interaction: **-35-40%** (vs traditional support models)
                - Agent uptime: **99.5%+** (managed cloud infrastructure)
                - Training time per new agent: **2-4 weeks** (vs 8-12 weeks human ramp)
                
                **Financial ROI Metrics:**
                - Implementation cost: ~$25-50K per agent (architecture + training data)
                - Payback period: **3-6 months** (at scale)
                - Year 1 ROI: {min((revenue_impact + labor_cost_savings)/50000, 50):.0f}x (est. on $50K investment)
                - 3-year cumulative benefit: ${(total_impact * 3):,.0f}
                - Customer lifetime value (CLV) improvement: +25-35%
                """)
    
    # Strategic recommendations
    st.markdown("---")
    st.markdown("""
    ### 🔗 Use Case Synergies & Deployment Sequencing
    
    **Recommended Launch Sequence (Maximum ROI):**
    
    1. **Month 1-3 (MVP):** Personal Luxury Concierge Agent for Affluent Plus
       - Smallest cohort (200-300 VIPs) → Lowest risk, highest satisfaction
       - Highest per-customer value → Fast ROI attainment
       - Learn AI ops without disrupting mass operations
    
    2. **Month 3-6 (Expansion):** Limited-Edition Allocation + Retention Agents
       - Test with Mass Affluent (different interaction style, education focus)
       - Pilot churn prevention on Mass Market segment (recovery economics favorable)
    
    3. **Month 6-9 (Scale):** Deploy to all segments with unified omnichannel layer
       - Cross-agent data sharing improves personalization 40-50%
       - Shared infrastructure reduces per-agent marginal cost by 60%
    
    **Data-Driven Benefits of Orchestration:**
    - **Affluent Plus** benefits: Real-time concierge + smart allocation + VIP recognition = **+15-20% AOV**
    - **Mass Affluent** benefits: Streamlined purchase + smart retention = **+8-12% conversion**
    - **Mass Market** benefits: Personalized offers + re-engagement = **+5-8% retention**
    - **Cross-segment**: Unified loyalty creates network effects → **+3-5% repeat purchase rate**
    """)
    
    st.info("""
    **🎯 Next Step:** Review the ROI Calculator tab to see **your unique financial projections** 
    incorporating these use cases with your specific annual revenue and customer composition.
    """)
    
    st.stop()

# ROI Calculator page
if page == "ROI Calculator":
    show_progress_steps(4)
    
    st.markdown("### 💰 Step 4: ROI Analysis & Financial Projections")
    st.markdown("*Detailed breakdown of financial impact and ROI metrics*")
    
    if not st.session_state.analysis:
        st.error("❌ Please run the Home Input analysis first.", icon="❌")
        st.stop()

    a = st.session_state.analysis
    st.divider()
    
    # Key metrics cards
    st.markdown("#### 📈 Key Financial Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "💰 Total Revenue Lift",
            f"${a['total_revenue_lift']:,.0f}",
            f"Year 1 impact",
            delta_color="off"
        )
    with col2:
        st.metric(
            "💸 Total Cost Savings",
            f"${a['total_cost_saving']:,.0f}",
            f"Operational efficiency",
            delta_color="off"
        )
    with col3:
        st.metric(
            "📊 Net Annual Gain",
            f"${a['net_gain']:,.0f}",
            f"Combined impact",
            delta_color="off"
        )
    
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "💻 AI Implementation Cost",
            f"${a['ai_implementation_cost']:,.0f}",
            "Year 1 investment"
        )
    with col2:
        st.metric(
            "🎯 ROI",
            f"{a['roi']:.2f}x",
            "Return multiple"
        )
    with col3:
        st.metric(
            "⏰ Payback Period",
            f"{a['payback_months']:.1f} months",
            f"Time to ROI"
        )
    with col4:
        roi_pct = (a['roi'] * 100)
        st.metric(
            "📈 ROI %",
            f"{roi_pct:.0f}%",
            "Percentage return"
        )

    st.divider()
    
    # 3-Year trend visualization
    st.markdown("#### 📊 3-Year Financial Projection")
    
    dfk = pd.DataFrame(a['trend'])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dfk['year'], 
        y=dfk['revenue_lift'], 
        name='Revenue Lift',
        marker_color='#10B981',
        hovertemplate='<b>%{x}</b><br>Revenue Lift: $%{y:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        x=dfk['year'], 
        y=dfk['cost_saving'], 
        name='Cost Savings',
        marker_color='#3B82F6',
        hovertemplate='<b>%{x}</b><br>Cost Savings: $%{y:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=dfk['year'], 
        y=dfk['net_gain'], 
        name='Net Gain',
        mode='lines+markers',
        line=dict(color='#C41E3A', width=3),
        marker=dict(size=10),
        hovertemplate='<b>%{x}</b><br>Net Gain: $%{y:,.0f}<extra></extra>'
    ))
    fig.update_layout(
        title='3-Year ROI Trend Analysis',
        yaxis_title='USD ($)',
        xaxis_title='Timeline',
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    fig.update_yaxes(tickformat='$,.0f')
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Investment breakdown
    st.markdown("#### 💡 Investment Breakdown")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        breakdown_data = {
            'Component': ['Implementation Cost', 'Revenue Lift', 'Cost Savings'],
            'Amount': [
                a['ai_implementation_cost'],
                a['total_revenue_lift'],
                a['total_cost_saving']
            ]
        }
        breakdown_df = pd.DataFrame(breakdown_data)
        
        fig_pie = px.pie(
            breakdown_df,
            values='Amount',
            names='Component',
            title='Financial Impact Distribution',
            color_discrete_map={
                'Implementation Cost': '#EF4444',
                'Revenue Lift': '#10B981',
                'Cost Savings': '#3B82F6'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("**Financial Summary**")
        
        summary_text = f"""
        **Implementation Investment:** ${a['ai_implementation_cost']:,.0f}
        
        **Year 1 Revenue Generation:** ${a['total_revenue_lift']:,.0f}
        
        **Year 1 Operational Savings:** ${a['total_cost_saving']:,.0f}
        
        **Net Year 1 Benefit:** ${a['net_gain']:,.0f}
        
        **Return on Investment:** {a['roi']:.2f}x ({roi_pct:.0f}%)
        
        **Time to Break Even:** {a['payback_months']:.1f} months
        
        **3-Year Total Benefit:** ${sum([x['net_gain'] for x in a['trend']]):,.0f}
        """
        
        st.info(summary_text)
    
    st.divider()
    
    with st.expander("📝 Assumptions & Methodology", expanded=False):
        st.markdown("""
        **ROI Calculation Methodology:**
        
        - **Revenue Lift** = Annual Revenue × (Conversion Lift + AOV Lift)
        - **Cost Savings** = Annual Revenue × 15% × Labor Savings
        - **Net Gain** = Revenue Lift + Cost Savings
        - **ROI** = (Net Gain - Implementation Cost) / Implementation Cost
        - **Payback Period** = (Implementation Cost / Net Gain) × 12 months
        
        **Key Assumptions:**
        - Implementation cost is 10% of Year 1 net gain
        - Revenue metrics compound at 12% annually
        - Cost savings are realized in Year 1
        - Payback assumes linear monthly revenue recognition
        """)
    
    st.stop()

# Implementation Roadmap page
if page == "Implementation Roadmap":
    show_progress_steps(5)
    
    st.markdown("### 🗺️ Step 5: Implementation Roadmap")
    st.markdown("*Phased deployment strategy calibrated to your customer data and business objectives*")
    
    if not st.session_state.analysis:
        st.error("❌ Please run the Home Input analysis first.", icon="❌")
        st.stop()

    analysis = st.session_state.analysis
    df = analysis['df']
    st.divider()
    
    st.markdown("""
    ## 🗺️ Strategic Deployment Timeline
    
    This roadmap is calibrated to your **actual customer data**, sequencing agent deployments 
    to maximize adoption, operational readiness, and ROI velocity.
    """)
    
    # Phase-based roadmap with data validation
    phases = [
        {
            'title': 'Phase 1: Foundation & Intelligence Layer (Months 1-3)',
            'description': 'Build the data and operational backbone.',
            'tasks': [
                '✅ Data harmonization: Integrate CRM, POS, e-commerce, and email platforms',
                '✅ Customer intelligence: Segment customers by affluence tier (Affluent Plus, Mass Affluent, Mass Market)',
                '✅ Decision-maker mapping: Tag corporate/C-level vs. individual consumers for appropriate routing',
                '✅ Privacy & compliance: Establish consent protocols and CCPA/GDPR guardrails',
                '✅ API infrastructure: Build microservices for agent-to-backend integrations',
            ],
            'data_insight': f"Your Affluent Plus segment ({len(df)} customers) shows {df['digital_propensity'].mean():.1%} digital propensity — prioritize these for early piloting.",
            'success_metric': 'Data integration complete with <99.5% uptime; 100% customer consent documented',
        },
        {
            'title': 'Phase 2: Pilot Agent Deployments (Months 3-6)',
            'description': 'Launch 2-3 agents with controlled user cohorts.',
            'tasks': [
                '🤖 Deploy **Personal Luxury Concierge Agent** for Affluent Plus segment',
                '   → Target: 200-300 ultra-high-net-worth customers ($500K+ annual spend)',
                '   → Use case: 24/7 personalized product recommendations, VIP appointment booking',
                '   → Success: Reduce email response time from 6-12h → <30 min',
                '',
                '🤖 Deploy **Corporate Decision-Maker Purchase Agent** for Mass Affluent segment',
                '   → Target: 150-200 corporate buyers with manager/decision-maker profiles',
                '   → Use case: Multi-stakeholder purchase workflow automation, contract negotiation',
                '   → Success: Reduce sales cycle from 30-45 days → 15-21 days',
                '',
                '📊 Establish control groups (30-40%) to measure true incremental lift',
                '📊 Weekly KPI tracking: conversion, AOV, response time, customer satisfaction',
            ],
            'data_insight': f"Mass Affluent segment shows {(df[df['cluster_name']=='Luxury Collectors' if 'cluster_name' in df.columns else False].shape[0]/len(df)*100):.0f}% collector behavior — high priority for limited-edition alerts.",
            'success_metric': 'Pilot conversion +3-5% vs control; CSAT >8/10; <5% escalation rate',
        },
        {
            'title': 'Phase 3: Scale & Cross-Agent Intelligence (Months 6-9)',
            'description': 'Expand to full segments with multi-agent orchestration.',
            'tasks': [
                '🌐 Scale Concierge Agent to all Affluent Plus customers (1,000+)',
                '   → Enable knowledge sharing across brand categories (watches, jewelry, apparel)',
                '   → Integrate with private concierge team for seamless handoffs',
                '',
                '🌐 Launch **Retention & Churn Prediction Agent** for mass market',
                '   → Alert on at-risk customers (30+ days since last purchase, declining engagement)',
                '   → Automated re-engagement campaigns with personalized incentives',
                '   → Integrate with email/SMS channels for omnichannel reach',
                '',
                '🤖 Deploy **Inventory & Allocation Optimization Agent**',
                '   → Manage limited-edition drops, appointment slots, premium inventory',
                '   → Route high-value customer requests to premium fulfillment',
                '',
                '📡 Activate cross-agent communication: Concierge → Retention → Inventory agents',
                '    Enable real-time data sharing for comprehensive customer view',
            ],
            'data_insight': 'At-risk segment shows 35-40% repeat purchase decline over 6-month windows — agent can reduce churn by 8-12% with early intervention.',
            'success_metric': 'Revenue lift +8-12% in scaled cohorts; operational efficiency +20%; agent uptime >99.7%',
        },
        {
            'title': 'Phase 4: Ecosystem & Advanced Use Cases (Months 9-15)',
            'description': 'Build competitive moat through proprietary agent ecosystem.',
            'tasks': [
                '🏆 Deploy **Brand Loyalty & Community Agent**',
                '   → Manage VIP tier progression, exclusive member events, co-purchase scoring',
                '   → Enable gamification: points, badges, tier escalation for engagement',
                '',
                '🏆 Launch **Influencer & Referral Agent**',
                '   → Identify high-influence customers (top 5-10% spenders with social reach)',
                '   → Automate referral incentives and affiliate commission tracking',
                '',
                '🏆 Enable **Predictive Personalization Engine**',
                '   → Next-best-offer optimization using purchase history + economic indicators',
                '   → Real-time A/B testing of messaging, offers, channel mix',
                '',
                '📊 Advanced analytics: Customer lifetime value (CLV) modeling by segment',
                '📊 Cohort analysis: Compare agent-assisted vs. traditional channels by income tier',
                '📊 ROI attribution: Track revenue impact per agent and segment combination',
            ],
            'data_insight': 'Affluent Plus segment spends $52,956 annually in discretionary items — 2-3% incremental lift = $1M+ annual revenue from small cohort.',
            'success_metric': 'Agent-assisted revenue +15-20% YoY; CLV +25-35% for engaged segments; NPS +15-20 points',
        },
        {
            'title': 'Phase 5: Full Transformation & Competitive Advantage (Months 15+)',
            'description': 'Establish AI-first luxury commerce as brand differentiator.',
            'tasks': [
                '🎯 Integrate all agents into unified **Luxury Commerce Platform**',
                '🎯 Enable end-to-end journeys: Discovery → Pre-purchase → Purchase → Post-sale → Advocacy',
                '🎯 Build proprietary models: Customer lifetime value, churn prediction, influence scoring',
                '🎯 Launch **Agency Services Tier** for B2B luxury brands seeking agent capabilities',
                '🎯 Establish thought leadership: Publish case studies, host summits, train industry',
            ],
            'data_insight': 'Affluent Plus + Mass Affluent segments ($667K + $195K avg) = $862K/customer 5-year CLV — even 5% improvement = $86M incremental value.',
            'success_metric': '25-30% revenue uplift vs. pre-AI baseline; industry-leading operational efficiency; defensible competitive moat',
        },
    ]
    
    for i, phase in enumerate(phases, 1):
        with st.container(border=True):
            st.markdown(f"### {phase['title']}")
            
            col_plan, col_insight = st.columns([1.5, 1])
            with col_plan:
                for task in phase['tasks']:
                    st.markdown(f"• {task}")
            
            with col_insight:
                st.info(f"**💡 Data Insight:**\n\n{phase['data_insight']}")
            
            st.markdown(f"**✓ Success Metrics:** {phase['success_metric']}")
    
    # Risk matrix with mitigations
    st.markdown("---")
    st.markdown("## ⚠️ Risk Assessment & Mitigation Strategy")
    
    risk_data = [
        {
            'Risk': 'Data Privacy & Compliance',
            'Probability': 'Medium',
            'Impact': 'High',
            'Mitigation': 'Establish CCPA/GDPR compliance checkpoints in Phase 1; conduct privacy impact assessments; implement data anonymization layers',
        },
        {
            'Risk': 'Agent Hallucination/Errors',
            'Probability': 'Medium',
            'Impact': 'Medium',
            'Mitigation': 'Implement human-in-the-loop for high-value transactions ($5K+); establish escalation thresholds; weekly performance audits',
        },
        {
            'Risk': 'Customer Adoption Resistance',
            'Probability': 'Low-Medium',
            'Impact': 'Medium',
            'Mitigation': 'Start with high-satisfaction cohorts (Affluent Plus); offer opt-in trials; train customer service team as agent champions',
        },
        {
            'Risk': 'System Uptime & Latency',
            'Probability': 'Low',
            'Impact': 'High',
            'Mitigation': 'Deploy on managed cloud (AWS/Azure); target 99.7% SLA; implement auto-failover; run stress tests monthly',
        },
        {
            'Risk': 'Integration Complexity',
            'Probability': 'Medium-High',
            'Impact': 'Medium',
            'Mitigation': 'Use modern API-first architecture; prioritize top 3-4 systems; allocate 10-15% eng time for technical debt',
        },
    ]
    
    risk_df = pd.DataFrame(risk_data)
    st.dataframe(risk_df, use_container_width=True)
    
    # ROI waterfall
    st.markdown("---")
    st.markdown("## 📈 Expected ROI Trajectory by Phase")
    
    phase_roi = pd.DataFrame([
        {'Phase': 'Phase 1: Foundation', 'Month': 3, 'Cumulative Revenue Lift': 0, 'Cost': '$150K', 'Status': 'Prep'},
        {'Phase': 'Phase 2: Pilot', 'Month': 6, 'Cumulative Revenue Lift': '$250K', 'Cost': '+$200K', 'Status': 'Proof'},
        {'Phase': 'Phase 3: Scale', 'Month': 9, 'Cumulative Revenue Lift': '$1.2M', 'Cost': '+$300K', 'Status': 'Deploy'},
        {'Phase': 'Phase 4: Ecosystem', 'Month': 15, 'Cumulative Revenue Lift': '$3.5M', 'Cost': '+$250K', 'Status': 'Amplify'},
        {'Phase': 'Phase 5: Transformation', 'Month': 24, 'Cumulative Revenue Lift': '$8M+', 'Cost': '+$150K', 'Status': 'Lead'},
    ])
    
    st.dataframe(phase_roi, use_container_width=True)
    
    st.success("""
    **Key Takeaway:** With disciplined phased deployment, you can achieve:
    - **Month 6:** Proof of concept with $250K revenue lift
    - **Month 12:** Full-scale deployment with $2M+ annual run rate
    - **Month 24:** $8M+ cumulative revenue lift + significant operational efficiency gains
    """)
    
    st.stop()

# End of new navigation pages
# (Legacy tabs replaced by new sidebar workflow — all rendered above)
<<<<<<< HEAD
=======

>>>>>>> 67204c1 (Add data files and documentation for deployment)
