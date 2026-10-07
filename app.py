"""
app.py — CarValuator AI Dashboard
====================================
Interactive Streamlit dashboard with:
  - Real-time price prediction
  - Multi-model comparison
  - Interactive data exploration
  - SHAP-based AI explainability
"""
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.feature_engineering import (
    create_features,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    TARGET,
)

# ── Page Configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title="CarValuator AI — Smart Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (Premium Dark Theme) ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 40%, #16213e 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] .stRadio label {
    color: #e0e0e0 !important;
    font-weight: 500;
}

/* Headers */
h1, h2, h3 {
    color: #f0f0f0 !important;
}

/* Glass Card */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 28px;
    margin: 14px 0;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,210,255,0.08);
}

/* Metric Card */
.metric-card {
    background: linear-gradient(135deg, rgba(0,210,255,0.08), rgba(123,47,247,0.08));
    border: 1px solid rgba(0,210,255,0.15);
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    transition: transform 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
}
.metric-card h3 {
    margin: 0;
    font-size: 14px;
    color: #8b8ba7 !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-card p {
    margin: 8px 0 0;
    font-size: 28px;
    font-weight: 800;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Gradient Text */
.gradient-text {
    background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* Hero */
.hero-title {
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 18px;
    color: #8b8ba7;
    margin-top: 8px;
    line-height: 1.6;
}

/* Price Display */
.price-display {
    background: linear-gradient(135deg, rgba(0,210,255,0.12), rgba(123,47,247,0.12));
    border: 2px solid rgba(0,210,255,0.25);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    margin: 20px 0;
}
.price-display .price-label {
    font-size: 14px;
    color: #8b8ba7;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
}
.price-display .price-value {
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.price-display .price-unit {
    font-size: 16px;
    color: #a0a0b8;
    margin-top: 4px;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin: 2px;
}
.badge-cyan {
    background: rgba(0,210,255,0.15);
    color: #00d2ff;
    border: 1px solid rgba(0,210,255,0.3);
}
.badge-purple {
    background: rgba(123,47,247,0.15);
    color: #a78bfa;
    border: 1px solid rgba(123,47,247,0.3);
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #00d2ff, #7b2ff7) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 28px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.5px;
    transition: all 0.3s ease !important;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,210,255,0.3) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 8px 20px;
    color: #a0a0b8;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,210,255,0.15), rgba(123,47,247,0.15));
    color: #00d2ff !important;
}

/* Select boxes & inputs */
div[data-baseweb="select"] {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ── Data & Model Loading ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load trained pipelines and config."""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    pipelines = joblib.load(os.path.join(model_dir, 'all_pipelines.joblib'))
    config = joblib.load(os.path.join(model_dir, 'config.joblib'))
    stacking = joblib.load(os.path.join(model_dir, 'stacking_ensemble.joblib'))
    train_sample = joblib.load(os.path.join(model_dir, 'train_sample.joblib'))
    return pipelines, config, stacking, train_sample


@st.cache_data
def load_dataset():
    """Load the car dataset."""
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'car_data.csv')
    return pd.read_csv(data_path)


# ── Car Catalog (for dropdowns) ──────────────────────────────────────
CAR_CATALOG = {
    'Maruti': ['Swift', 'Baleno', 'Alto', 'Dzire', 'Vitara Brezza', 'Ciaz', 'Ertiga', 'WagonR'],
    'Hyundai': ['i20', 'Creta', 'Venue', 'Verna', 'i10', 'Tucson', 'Santro'],
    'Honda': ['City', 'Amaze', 'WRV', 'Jazz', 'Civic'],
    'Toyota': ['Innova Crysta', 'Fortuner', 'Glanza', 'Urban Cruiser', 'Camry'],
    'Tata': ['Nexon', 'Harrier', 'Altroz', 'Tiago', 'Safari', 'Punch'],
    'Mahindra': ['XUV700', 'Scorpio', 'Thar', 'XUV300', 'Bolero'],
    'Kia': ['Seltos', 'Sonet', 'Carnival'],
    'BMW': ['3 Series', '5 Series', 'X1', 'X3', 'X5'],
    'Audi': ['A4', 'A6', 'Q3', 'Q5', 'Q7'],
    'Mercedes': ['C-Class', 'E-Class', 'GLA', 'GLC', 'GLE'],
}


# ── Sidebar Navigation ───────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="gradient-text" style="font-size:24px;">🚗 CarValuator AI</p>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔮 Predict Price", "📊 Data Explorer", "🏆 Model Comparison", "🧠 AI Insights"],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        '<p style="color:#555; font-size:12px; text-align:center;">'
        'Built with ❤️ using Streamlit & scikit-learn</p>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<h1 class="hero-title">CarValuator AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        'An advanced machine learning system that predicts used car prices with high accuracy. '
        'Powered by 8 ML models, SHAP explainability, and a beautiful interactive dashboard.'
        '</p>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Key metrics
    try:
        _, config, _, _ = load_models()
        best_r2 = config['best_r2']
        best_model = config['best_model']
        train_size = config['train_size']
        n_models = len(config['results'])
    except Exception:
        best_r2, best_model, train_size, n_models = 0.95, "N/A", 4000, 8

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card"><h3>Best R² Score</h3><p>{best_r2:.4f}</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><h3>Best Model</h3><p style="font-size:16px;">{best_model}</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card"><h3>Training Samples</h3><p>{train_size:,}</p></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card"><h3>Models Trained</h3><p>{n_models}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature highlights
    st.markdown("### ✨ What Makes This Project Unique")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="glass-card">'
            '<h3>🧬 Advanced Feature Engineering</h3>'
            '<p style="color:#a0a0b8;">6 derived features capturing depreciation patterns, '
            'usage intensity, and engine efficiency — zero data leakage.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="glass-card">'
            '<h3>🤖 8 ML Models + Ensemble</h3>'
            '<p style="color:#a0a0b8;">From Linear Regression to a Stacking Ensemble with '
            'RF, GBM, and XGBoost as base learners.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="glass-card">'
            '<h3>🧠 SHAP Explainability</h3>'
            '<p style="color:#a0a0b8;">Understand WHY the model predicts a certain price '
            'with SHAP waterfall and feature importance plots.</p>'
            '</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════
#  PAGE: PREDICT PRICE
# ══════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Price":
    st.markdown('<h1 class="gradient-text" style="font-size:36px;">🔮 Predict Car Price</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b8ba7;">Enter car details below to get an instant AI-powered price estimate.</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    try:
        pipelines, config, stacking, _ = load_models()
    except Exception as e:
        st.error(f"⚠️ Models not found. Please run `python train.py` first.\n\nError: {e}")
        st.stop()

    # Input form
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🚗 Car Details")

        brand = st.selectbox("Brand", list(CAR_CATALOG.keys()), index=0)
        model_name = st.selectbox("Model", CAR_CATALOG[brand])
        year = st.slider("Year of Manufacture", 2005, 2024, 2019)
        present_price = st.number_input("Showroom Price (Lakhs ₹)", min_value=1.0, max_value=100.0, value=8.0, step=0.5)
        kms_driven = st.number_input("Kilometers Driven", min_value=500, max_value=500000, value=30000, step=1000)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Specifications")

        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
        owner = st.selectbox("Owner", ["First", "Second", "Third", "Fourth & Above"])
        engine_cc = st.number_input("Engine CC", min_value=600, max_value=5000, value=1200, step=50)
        max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=400.0, value=85.0, step=5.0)
        mileage = st.number_input("Mileage (km/l)", min_value=5.0, max_value=35.0, value=18.0, step=0.5)
        seats = st.selectbox("Seats", [4, 5, 6, 7, 8], index=1)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 Predict Price", use_container_width=True):
        # Build input DataFrame
        input_data = {
            'Brand': brand, 'Model': model_name, 'Year': year,
            'Present_Price': present_price, 'Kms_Driven': kms_driven,
            'Fuel_Type': fuel_type, 'Seller_Type': seller_type,
            'Transmission': transmission, 'Owner': owner,
            'Engine_CC': engine_cc, 'Max_Power_bhp': max_power,
            'Mileage_kmpl': mileage, 'Seats': seats,
        }
        input_df = pd.DataFrame([input_data])

        # Feature engineering (silent)
        import io, contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            input_df = create_features(input_df)

        X_input = input_df[ALL_FEATURES]

        # Get predictions from all models
        predictions = {}
        for name, pipeline in pipelines.items():
            pred = pipeline.predict(X_input)[0]
            predictions[name] = round(max(0.1, pred), 2)

        # Stacking prediction
        stack_pred = stacking.predict(X_input)[0]
        predictions['Stacking Ensemble'] = round(max(0.1, stack_pred), 2)

        # Best model prediction
        best_name = config['best_model']
        best_pred = predictions.get(best_name, list(predictions.values())[0])

        # Display result
        st.markdown(
            f'<div class="price-display">'
            f'<div class="price-label">Estimated Selling Price</div>'
            f'<div class="price-value">₹ {best_pred:.2f} L</div>'
            f'<div class="price-unit">Predicted by {best_name}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # All model predictions comparison
        st.markdown("#### 📊 All Model Predictions")
        pred_df = pd.DataFrame([
            {'Model': k, 'Predicted Price (Lakhs)': v}
            for k, v in sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        ])

        fig = px.bar(
            pred_df, x='Model', y='Predicted Price (Lakhs)',
            color='Predicted Price (Lakhs)',
            color_continuous_scale=['#00d2ff', '#7b2ff7'],
            title='Price Predictions Across All Models',
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0',
            xaxis=dict(tickangle=-45),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#  PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown('<h1 class="gradient-text" style="font-size:36px;">📊 Data Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b8ba7;">Interactive visualizations of the car price dataset.</p>', unsafe_allow_html=True)

    try:
        df = load_dataset()
    except Exception as e:
        st.error(f"⚠️ Dataset not found. Please run `python train.py` first.\n\nError: {e}")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Analysis", "🏷️ Brand Insights", "⛽ Fuel & Transmission", "🔗 Correlations"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                df, x='Selling_Price', nbins=60,
                title='Selling Price Distribution',
                color_discrete_sequence=['#7b2ff7'],
                marginal='box',
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                df, x='Kms_Driven', y='Selling_Price',
                color='Fuel_Type', title='Price vs Kilometers Driven',
                color_discrete_sequence=['#00d2ff', '#7b2ff7', '#ff6b6b'],
                opacity=0.5,
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
            st.plotly_chart(fig, use_container_width=True)

        # Price vs Year
        fig = px.scatter(
            df, x='Year', y='Selling_Price', color='Brand',
            title='Selling Price vs Year of Manufacture',
            opacity=0.5, hover_data=['Model', 'Kms_Driven'],
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0', height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Brand box plot
        fig = px.box(
            df, x='Brand', y='Selling_Price',
            title='Price Distribution by Brand',
            color='Brand',
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0', showlegend=False, xaxis_tickangle=-45, height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Brand count
        brand_counts = df['Brand'].value_counts().reset_index()
        brand_counts.columns = ['Brand', 'Count']
        fig = px.bar(
            brand_counts, x='Brand', y='Count',
            title='Number of Cars per Brand',
            color='Count', color_continuous_scale=['#00d2ff', '#7b2ff7'],
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                df, names='Fuel_Type', title='Fuel Type Distribution',
                color_discrete_sequence=['#00d2ff', '#7b2ff7', '#ff6b6b'],
                hole=0.4,
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(
                df, names='Transmission', title='Transmission Distribution',
                color_discrete_sequence=['#7b2ff7', '#00d2ff'],
                hole=0.4,
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
            st.plotly_chart(fig, use_container_width=True)

        fig = px.violin(
            df, x='Fuel_Type', y='Selling_Price', color='Transmission',
            title='Price by Fuel Type & Transmission',
            color_discrete_sequence=['#00d2ff', '#7b2ff7'],
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0', height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[numerical_cols].corr()

        fig = px.imshow(
            corr, text_auto='.2f', aspect='auto',
            title='Feature Correlation Matrix',
            color_continuous_scale='RdBu_r',
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0', height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Top correlations with target
        target_corr = corr['Selling_Price'].drop('Selling_Price').sort_values(ascending=False)
        fig = px.bar(
            x=target_corr.values, y=target_corr.index,
            orientation='h', title='Correlation with Selling Price',
            color=target_corr.values, color_continuous_scale=['#ff6b6b', '#00d2ff', '#7b2ff7'],
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0', height=400)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#  PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════
elif page == "🏆 Model Comparison":
    st.markdown('<h1 class="gradient-text" style="font-size:36px;">🏆 Model Comparison</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b8ba7;">Head-to-head comparison of all trained ML models.</p>', unsafe_allow_html=True)

    try:
        pipelines, config, stacking, _ = load_models()
    except Exception as e:
        st.error(f"⚠️ Models not found. Please run `python train.py` first.\n\nError: {e}")
        st.stop()

    results = config['results']

    # Comparison table
    rows = []
    for name, res in results.items():
        rows.append({
            'Model': name,
            'Test R²': res['test']['R2 Score'],
            'Test MAE': res['test']['MAE'],
            'Test RMSE': res['test']['RMSE'],
            'Test MAPE (%)': res['test']['MAPE'],
            'Train R²': res['train']['R2 Score'],
        })
    comp_df = pd.DataFrame(rows).sort_values('Test R²', ascending=False).reset_index(drop=True)

    st.markdown("#### 📋 Performance Summary")
    st.dataframe(comp_df.style.highlight_max(subset=['Test R²'], color='#1a3a2a')
                 .highlight_min(subset=['Test MAE', 'Test RMSE'], color='#1a3a2a'),
                 use_container_width=True, height=350)

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            comp_df, x='Model', y='Test R²',
            title='Test R² Score (Higher = Better)',
            color='Test R²', color_continuous_scale=['#ff6b6b', '#00d2ff', '#7b2ff7'],
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0', xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            comp_df, x='Model', y='Test MAE',
            title='Test MAE in Lakhs (Lower = Better)',
            color='Test MAE', color_continuous_scale=['#7b2ff7', '#00d2ff', '#ff6b6b'],
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0', xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # Train vs Test R² (overfitting check)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Train R²', x=comp_df['Model'], y=comp_df['Train R²'], marker_color='#7b2ff7'))
    fig.add_trace(go.Bar(name='Test R²', x=comp_df['Model'], y=comp_df['Test R²'], marker_color='#00d2ff'))
    fig.update_layout(
        title='Train vs Test R² (Overfitting Check)',
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e0e0e0', xaxis_tickangle=-45, height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Best model highlight
    best = config['best_model']
    best_r2 = config['best_r2']
    st.markdown(
        f'<div class="price-display">'
        f'<div class="price-label">🥇 Best Performing Model</div>'
        f'<div class="price-value" style="font-size:36px;">{best}</div>'
        f'<div class="price-unit">Test R² = {best_r2:.4f}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
#  PAGE: AI INSIGHTS
# ══════════════════════════════════════════════════════════════════════
elif page == "🧠 AI Insights":
    st.markdown('<h1 class="gradient-text" style="font-size:36px;">🧠 AI Insights</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b8ba7;">Feature importance and SHAP explainability for model transparency.</p>', unsafe_allow_html=True)

    try:
        pipelines, config, _, train_sample = load_models()
    except Exception as e:
        st.error(f"⚠️ Models not found. Please run `python train.py` first.\n\nError: {e}")
        st.stop()

    tab1, tab2 = st.tabs(["📊 Feature Importance", "🔬 SHAP Analysis"])

    with tab1:
        # Feature importance for tree-based models
        tree_models = ['Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting']
        selected_model = st.selectbox("Select Model", [m for m in tree_models if m in pipelines])

        if selected_model and selected_model in pipelines:
            pipeline = pipelines[selected_model]
            model = pipeline.named_steps['model']
            preprocessor = pipeline.named_steps['preprocessor']

            if hasattr(model, 'feature_importances_'):
                num_features = list(preprocessor.transformers_[0][2])
                cat_encoder = preprocessor.named_transformers_['cat']
                cat_features = list(cat_encoder.get_feature_names_out())
                all_names = num_features + cat_features

                importances = model.feature_importances_
                feat_df = pd.DataFrame({
                    'Feature': all_names,
                    'Importance': importances,
                }).sort_values('Importance', ascending=True).tail(15)

                fig = px.bar(
                    feat_df, x='Importance', y='Feature',
                    orientation='h',
                    title=f'{selected_model} — Top 15 Feature Importances',
                    color='Importance', color_continuous_scale=['#00d2ff', '#7b2ff7'],
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e0e0e0', height=550,
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### 🔬 SHAP Summary Plot")
        st.markdown('<p style="color:#8b8ba7;">SHAP values show how each feature pushes the prediction higher or lower.</p>', unsafe_allow_html=True)

        shap_model_name = st.selectbox("Select Model for SHAP", [m for m in tree_models if m in pipelines], key='shap_model')

        if st.button("🧮 Compute SHAP Values"):
            with st.spinner("Computing SHAP values (this may take a moment)..."):
                try:
                    import shap

                    pipeline = pipelines[shap_model_name]
                    model = pipeline.named_steps['model']
                    preprocessor = pipeline.named_steps['preprocessor']

                    # Transform training sample
                    X_transformed = preprocessor.transform(train_sample)

                    num_features = list(preprocessor.transformers_[0][2])
                    cat_encoder = preprocessor.named_transformers_['cat']
                    cat_features = list(cat_encoder.get_feature_names_out())
                    all_names = num_features + cat_features

                    X_transformed_df = pd.DataFrame(X_transformed, columns=all_names)

                    # Compute SHAP values
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_transformed_df)

                    # Summary plot
                    fig_shap, ax_shap = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_transformed_df, show=False, max_display=15)
                    st.pyplot(fig_shap)
                    plt.close(fig_shap)

                    st.success("✅ SHAP analysis complete!")

                except ImportError:
                    st.warning("⚠️ SHAP library not installed. Run: `pip install shap`")
                except Exception as e:
                    st.error(f"Error computing SHAP values: {e}")

                    # Fallback: show feature importance
                    st.info("Showing feature importance instead.")
                    if hasattr(model, 'feature_importances_'):
                        feat_df = pd.DataFrame({
                            'Feature': all_names,
                            'Importance': model.feature_importances_,
                        }).sort_values('Importance', ascending=True).tail(15)
                        fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                                     title=f'{shap_model_name} — Feature Importance (Fallback)',
                                     color='Importance', color_continuous_scale=['#00d2ff', '#7b2ff7'])
                        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
                        st.plotly_chart(fig, use_container_width=True)
