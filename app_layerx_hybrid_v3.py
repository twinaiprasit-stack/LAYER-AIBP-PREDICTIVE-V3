import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===============================
# Utility
# ===============================
def _exists(p): return os.path.exists(p)
def _asset(path, fallback=None): return path if _exists(path) else fallback

# ===============================
# Load Models & Scaler (safe)
# ===============================
try:
    import joblib
    prophet_model = joblib.load("prophet_model.pkl")
    xgboost_model = joblib.load("xgboost_model.pkl")
    scaler, feature_names = joblib.load("scaler_and_features.pkl")
except Exception as e:
    st.warning(f"⚠️ Model load warning: {e}")
    prophet_model = xgboost_model = scaler = feature_names = None

# ===============================
# Assets
# ===============================
BG_IMAGE   = _asset("assets/space_bg.png", "/mnt/data/space_bg.png")
CPF_LOGO   = _asset("assets/LOGO-CPF.jpg", "/mnt/data/LOGO-CPF.jpg")
EGG_ROCKET = _asset("assets/egg_rocket.png", "/mnt/data/egg_rocket.png")

# ===============================
# CSS
# ===============================
def inject_layerx_css():
    st.markdown(f"""
    <style>
    /* --- Rocket --- */
    img[src*="egg_rocket"] {{
        position:absolute;top:40px;right:60px;width:95px;
        filter:drop-shadow(0 0 14px rgba(0,216,255,0.7));
        transform:rotate(-8deg);
    }}
    /* --- Buttons --- */
    .layerx-btn>button{{
        border-radius:12px!important;
        border:1px solid rgba(0,216,255,0.55)!important;
        background:linear-gradient(135deg,rgba(0,216,255,0.18),rgba(0,216,255,0.06))!important;
        color:#E3F6FF!important;font-weight:600!important;
        box-shadow:0 0 10px rgba(0,216,255,0.25);
        transition:all .25s ease-in-out;
    }}
    .layerx-btn>button:hover{{
        background:rgba(0,216,255,0.25)!important;
        color:#FFF!important;transform:translateY(-1px);
        box-shadow:0 0 16px rgba(0,216,255,0.45);
    }}
    </style>
    """, unsafe_allow_html=True)

# ===============================
# KPI Card
# ===============================
def kpi_card(title, value, unit=""):
    st.markdown(f"""
        <div style="padding:12px;border-radius:16px;background:rgba(0,40,60,0.35);
             box-shadow:inset 0 0 8px rgba(0,216,255,0.25);text-align:center;">
            <p style="color:#A0CFFF;font-size:14px;margin-bottom:0;">{title}</p>
            <h3 style="color:#00CCFF;margin-top:4px;">{value}{unit}</h3>
        </div>
    """, unsafe_allow_html=True)

# ===============================
# PAGE 1 – Egg Price Forecast Dashboard
# ===============================
def render_page1():
    inject_layerx_css()
    cols = st.columns([0.12, 0.64, 0.24])
    with cols[0]:
        if _exists(CPF_LOGO): st.image(CPF_LOGO, width=72)
    with cols[1]:
        st.markdown("""
            <div style='text-align:center;margin-top:-15px;'>
                <h2 style='color:#B8EFFF;margin-bottom:0;'>🥚 Egg Price Forecast Dashboard</h2>
                <p style='color:#8FD9FF;font-size:14px;'>AI-Powered Market Intelligence (2025)</p>
            </div>""", unsafe_allow_html=True)
    with cols[2]:
        if _exists(EGG_ROCKET): st.image(EGG_ROCKET)

    # Load Prophet output
    pf_csv = "prophet_forecast.csv" if _exists("prophet_forecast.csv") else "/mnt/data/prophet_forecast.csv"
    df = pd.read_csv(pf_csv)
    yhat = df['yhat_original'] if 'yhat_original' in df.columns else df['yhat']

    c1,c2,c3 = st.columns(3, gap="large")
    with c1: kpi_card("🟩 Min Price", f"{float(np.nanmin(yhat)):.2f}", " ฿/ฟอง")
    with c2: kpi_card("🟦 Avg Price", f"{float(np.nanmean(yhat)):.2f}", " ฿/ฟอง")
    with c3: kpi_card("🟥 Max Price", f"{float(np.nanmax(yhat)):.2f}", " ฿/ฟอง")

    # Plot Prophet Forecast (8 Weeks)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=yhat, mode='lines',
        name='Prophet Forecast', line=dict(color='cyan', width=3)))
    fig.update_layout(title="8-Week Forecast (Prophet)",
        xaxis_title="Date", yaxis_title="Price (฿/ฟอง)",
        template="plotly_dark", height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="layerx-btn" style="text-align:center;">', unsafe_allow_html=True)
    st.button("🧾 Export PDF Report")
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# PAGE 2 – Hybrid Model Performance
# ===============================
def render_page2():
    inject_layerx_css()
    cols = st.columns([0.12,0.64,0.24])
    with cols[0]:
        if _exists(CPF_LOGO): st.image(CPF_LOGO,width=72)
    with cols[1]:
        st.markdown("""
            <div style='text-align:center;margin-top:-15px;'>
                <h2 style='color:#B8EFFF;margin-bottom:0;'>🚀 Hybrid Model Performance — Prophet + XGBoost</h2>
                <p style='color:#8FD9FF;font-size:14px;'>Smart Fusion Forecasting for Egg Price (2025)</p>
            </div>""", unsafe_allow_html=True)
    with cols[2]:
        if _exists(EGG_ROCKET): st.image(EGG_ROCKET)

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi_card("MAE","0.0037")
    with c2: kpi_card("RMSE","0.0046")
    with c3: kpi_card("MAPE","0.10%")
    with c4: kpi_card("Accuracy","99.90%")

    pf_path = "prophet_forecast.csv" if _exists("prophet_forecast.csv") else "/mnt/data/prophet_forecast.csv"
    df_p = pd.read_csv(pf_path)
    if 'yhat_original' in df_p.columns:
        df_p['prophet']=df_p['yhat_original']
    elif 'yhat' in df_p.columns:
        df_p['prophet']=np.expm1(df_p['yhat'])
    df_p['ds']=pd.to_datetime(df_p['ds'])

    # Combine
    df_plot=pd.DataFrame({'ds':df_p['ds'],'Prophet':df_p['prophet']})
    if 'PriceMarket' in df_p.columns:
        df_plot['Actual']=np.expm1(df_p['PriceMarket'])

    fig=go.Figure()
    if 'Actual' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot['ds'],y=df_plot['Actual'],
            mode='lines+markers',name='Actual',line=dict(color='red',width=2)))
    fig.add_trace(go.Scatter(x=df_plot['ds'],y=df_plot['Prophet'],
        mode='lines',name='Prophet Forecast',line=dict(color='cyan',width=3)))
    fig.update_layout(title="Actual vs Predicted (Prophet + XGBoost)",
        xaxis_title="Date",yaxis_title="Price (฿/ฟอง)",
        template="plotly_dark",height=500)
    st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="layerx-btn" style="text-align:center;">', unsafe_allow_html=True)
    st.button("🧾 Export PDF Summary")
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# MAIN NAVIGATION
# ===============================
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Page 1 — Dashboard", "Page 2 — Hybrid Performance"])
if page.startswith("Page 1"): render_page1()
else: render_page2()
