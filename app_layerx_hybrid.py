
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Egg Price Forecast | Layer X", page_icon="🥚", layout="wide")

# ---------- THEME / ASSETS ----------
BG_IMAGE = "/mnt/data/7cc2db54-4b0f-4179-9fd0-4e0411da902c.png"
CPF_LOGO = "/mnt/data/LOGO-CPF.jpg"
EGG_ROCKET = "/mnt/data/8e6f79d9-6091-482e-ba98-9cc2d78f85fe.png"

PRIMARY_CYAN = "#00d8ff"
NAVY = "#081b29"
AMBER = "#ffae42"
FONT_COLOR = "#E3F6FF"

def inject_layerx_css():
    bg_url = BG_IMAGE if os.path.exists(BG_IMAGE) else ""
    css = f"""
    <style>
    .stApp {{
        background: radial-gradient(ellipse at bottom left, rgba(0,216,255,0.08), rgba(0,0,0,0.2)), 
                    linear-gradient(180deg, rgba(8,27,41,0.95) 0%, rgba(3,10,17,0.98) 100%);
        color: {FONT_COLOR};
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}
    /* Space background overlay */
    .space-bg {{
        position: fixed;
        inset: 0;
        background-image: url('{bg_url}');
        background-size: cover;
        background-position: center;
        opacity: 0.25;
        filter: saturate(120%);
        z-index: -1;
    }}
    /* Header */
    .layerx-header {{
        backdrop-filter: blur(8px);
        background: linear-gradient(135deg, rgba(0,216,255,0.08), rgba(255,255,255,0.02));
        padding: 16px 20px;
        border-radius: 16px;
        border: 1px solid rgba(0,216,255,0.25);
        box-shadow: 0 8px 24px rgba(0,216,255,0.12);
    }}
    .title {{
        font-size: 28px; font-weight: 700; letter-spacing: 0.3px; color: {FONT_COLOR};
    }}
    .subtitle {{
        font-size: 14px; opacity: 0.85; margin-top: 4px;
    }}
    /* KPI cards */
    .kpi-card {{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(0,216,255,0.18);
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: inset 0 0 32px rgba(0,216,255,0.06), 0 10px 24px rgba(0,0,0,0.2);
    }}
    .kpi-label {{ font-size: 13px; opacity: 0.85; }}
    .kpi-value {{ font-size: 28px; font-weight: 700; color: {PRIMARY_CYAN}; }}
    /* Buttons */
    .layerx-btn > button {{
        border-radius: 12px !important;
        border: 1px solid rgba(0,216,255,0.45) !important;
        background: rgba(0,216,255,0.08) !important;
        color: {FONT_COLOR} !important;
        box-shadow: 0 10px 24px rgba(0,216,255,0.12) !important;
    }}
    /* Tabs underline */
    div[role="tablist"] > div[aria-selected="true"] {{
        border-bottom: 2px solid {PRIMARY_CYAN};
    }}
    </style>
    <div class="space-bg"></div>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_layerx_css()

# ---------- UTILITIES ----------
def feature_engineer_v2(df):
    # Ensure Date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.set_index('Date')
    df = df.sort_index()
    # Resample to weekly
    df_weekly = df.resample('W').mean()
    cols = ['PriceMarket', 'Forecast', 'Quota', 'Stock', 'FeedPrice']
    for c in cols:
        if c in df_weekly.columns:
            df_weekly[f'{c}_lag1'] = df_weekly[c].shift(1)
            df_weekly[f'{c}_lag2'] = df_weekly[c].shift(2)
            df_weekly[f'{c}_lag3'] = df_weekly[c].shift(3)
            df_weekly[f'{c}_rolling_mean_4'] = df_weekly[c].rolling(4).mean()
            df_weekly[f'{c}_rolling_std_4'] = df_weekly[c].rolling(4).std()
    df_weekly = df_weekly.interpolate(method='time').ffill().bfill()
    return df_weekly

@st.cache_resource(show_spinner=False)
def load_artifacts():
    prophet, xgb, scaler, feats = None, None, None, None
    try:
        prophet = joblib.load("/mnt/data/prophet_model.pkl")
    except Exception as e:
        st.sidebar.warning(f"Prophet model not found or cannot load: {e}")
    try:
        xgb = joblib.load("/mnt/data/xgboost_model.pkl")
    except Exception as e:
        st.sidebar.warning(f"XGBoost model not found or cannot load: {e}")
    try:
        d = joblib.load("/mnt/data/scaler_and_features.pkl")
        scaler, feats = d.get("scaler", None), d.get("feature_names", None)
    except Exception as e:
        st.sidebar.warning(f"Scaler/feature_names not found: {e}")
    return prophet, xgb, scaler, feats

prophet_model, xgboost_model, scaler, feature_names = load_artifacts()

# Fixed metrics from user requirement
FIXED_MAE = 0.0037
FIXED_RMSE = 0.0046
FIXED_MAPE = 0.10
FIXED_ACC = 99.90

def kpi_card(label, value, unit=""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}{unit}</div>
    </div>
    """, unsafe_allow_html=True)

def render_header(title, subtitle):
    cols = st.columns([0.12,0.6,0.28])
    with cols[0]:
        if os.path.exists(CPF_LOGO):
            st.image(CPF_LOGO, width=72)
    with cols[1]:
        st.markdown(f'<div class="layerx-header"><div class="title">{title}</div><div class="subtitle">{subtitle}</div></div>', unsafe_allow_html=True)
    with cols[2]:
        if os.path.exists(EGG_ROCKET):
            st.image(EGG_ROCKET, width=96)

def plot_forecast(ds, yhat, ylow=None, yhigh=None, actual_df=None, title="52-Week Forecast"):
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(ds, yhat, label='Prophet Forecast', linewidth=2)
    if ylow is not None and yhigh is not None:
        ax.fill_between(ds, ylow, yhigh, alpha=0.15, label='Confidence')
    if actual_df is not None and 'PriceMarket' in actual_df.columns:
        ax.scatter(actual_df.index, actual_df['PriceMarket'], s=12, c='red', label='Actual')
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig

def plot_hybrid_chart(df_plot):
    # expects columns: ds, prophet, xgb, hybrid, actual (some may be missing)
    fig, ax = plt.subplots(figsize=(12,5))
    if 'prophet' in df_plot: ax.plot(df_plot['ds'], df_plot['prophet'], label='Prophet', linewidth=2)
    if 'xgb' in df_plot: ax.plot(df_plot['ds'], df_plot['xgb'], label='XGBoost', linewidth=2)
    if 'hybrid' in df_plot: ax.plot(df_plot['ds'], df_plot['hybrid'], label='Hybrid', linewidth=2)
    if 'actual' in df_plot: ax.scatter(df_plot['ds'], df_plot['actual'], s=12, c='red', label='Actual')
    ax.set_title("Actual vs Hybrid Forecast")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig

def export_pdf(kpis, figs, filename="/mnt/data/LayerX_Hybrid_Report.pdf"):
    with PdfPages(filename) as pdf:
        # cover
        fig_cover = plt.figure(figsize=(11.69, 8.27))
        plt.axis('off')
        plt.text(0.05, 0.85, "Egg Price Forecast — Layer X (CPF Edition)", fontsize=20, weight='bold', color='#00ffff')
        plt.text(0.05, 0.77, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=11, color='#dddddd')
        # KPIs
        y = 0.62
        for label, value in kpis:
            plt.text(0.07, y, f"{label}: {value}", fontsize=14, color='#E3F6FF')
            y -= 0.06
        # add figures
        pdf.savefig(fig_cover); plt.close(fig_cover)
        for f in figs:
            pdf.savefig(f); plt.close(f)
    return filename

# ---------- SIDEBAR NAV ----------
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Go to", ["Page 1 — Dashboard", "Page 2 — Hybrid Performance"], label_visibility="collapsed")

# ---------- PAGE 1 ----------
if page.startswith("Page 1"):
    render_header("🥚 Egg Price Forecast Dashboard", "AI-Powered Market Intelligence (Layer X)")

    with st.expander("📤 Upload historical CSV (Date, PriceMarket, ...)", expanded=False):
        up_hist = st.file_uploader("Upload CSV for Prophet forecast (optional)", type="csv", key="hist_1")
    df_forecast_display = None
    fig1 = None

    prophet_csv_path = "/mnt/data/prophet_forecast.csv"
    if up_hist is not None:
        try:
            df_hist = pd.read_csv(up_hist)
            _ = feature_engineer_v2(df_hist.copy())
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif os.path.exists(prophet_csv_path):
        try:
            df_forecast_display = pd.read_csv(prophet_csv_path)
        except Exception as e:
            st.warning(f"Could not read prophet_forecast.csv: {e}")

    # If prophet_forecast.csv available, show KPIs and chart
    if df_forecast_display is not None:
        ds = pd.to_datetime(df_forecast_display['ds'])
        # figure out scale
        if 'yhat_original' in df_forecast_display.columns:
            yhat = df_forecast_display['yhat_original']
            ylow = df_forecast_display.get('yhat_lower_original', None)
            yhigh = df_forecast_display.get('yhat_upper_original', None)
        else:
            yhat = np.expm1(df_forecast_display.get('yhat', pd.Series([np.nan]*len(ds))))
            ylow = np.expm1(df_forecast_display.get('yhat_lower', pd.Series([np.nan]*len(ds))))
            yhigh = np.expm1(df_forecast_display.get('yhat_upper', pd.Series([np.nan]*len(ds))))
        # KPIs
        k1, k2, k3 = st.columns(3, gap="large")
        with k1: kpi_card("Min Price", f"{float(np.nanmin(yhat)):.2f}", " ฿/ฟอง")
        with k2: kpi_card("Avg Price", f"{float(np.nanmean(yhat)):.2f}", " ฿/ฟอง")
        with k3: kpi_card("Max Price", f"{float(np.nanmax(yhat)):.2f}", " ฿/ฟอง")
        fig1 = plot_forecast(ds, yhat, ylow, yhigh, actual_df=None, title="52-Week Forecast")
        st.pyplot(fig1)

        # Downloads / Export
        cA, cB = st.columns(2)
        with cA:
            st.download_button("⬇ Download Forecast CSV", data=df_forecast_display.to_csv(index=False).encode('utf-8'),
                               file_name="prophet_forecast_52w.csv", mime="text/csv")
        with cB:
            if st.button("🧾 Export PDF (Page 1)"):
                kpis = [("Min Price", f"{float(np.nanmin(yhat)):.2f} ฿/ฟอง"),
                        ("Avg Price", f"{float(np.nanmean(yhat)):.2f} ฿/ฟอง"),
                        ("Max Price", f"{float(np.nanmax(yhat)):.2f} ฿/ฟอง")]
                out = export_pdf(kpis, [fig1], "/mnt/data/LayerX_Report_Page1.pdf")
                st.success(f"PDF saved: {out}")
                st.markdown(f"[📥 Download PDF]({out})")
    else:
        st.info("Upload a CSV or provide prophet_forecast.csv to render the forecast.")

# ---------- PAGE 2 ----------
else:
    render_header("🚀 Hybrid Model Performance — Prophet + XGBoost", "Smart Fusion Forecasting (Layer X)")

    # Metric band (fixed per requirement)
    st.markdown("#### Hybrid Performance (fixed metrics)")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("MAE", f"{FIXED_MAE:.4f}")
    with c2: kpi_card("RMSE", f"{FIXED_RMSE:.4f}")
    with c3: kpi_card("MAPE", f"{FIXED_MAPE:.2f}%")
    with c4: kpi_card("Accuracy", f"{FIXED_ACC:.2f}%")

    up2 = st.file_uploader("📤 Upload CSV (Date, PriceMarket, Forecast, Quota, Stock, FeedPrice ...)", type="csv", key="hybrid_csv")
    df_plot = None
    hybrid_fig = None

    if up2 is not None:
        try:
            raw = pd.read_csv(up2)
            df_proc = feature_engineer_v2(raw.copy())

            # Prophet forecast source: use prophet_forecast.csv if exists
            df_prophet = None
            pf_path = "/mnt/data/prophet_forecast.csv"
            if os.path.exists(pf_path):
                df_prophet = pd.read_csv(pf_path)
                if 'yhat_original' in df_prophet.columns:
                    df_prophet['prophet'] = df_prophet['yhat_original']
                elif 'yhat' in df_prophet.columns:
                    df_prophet['prophet'] = np.expm1(df_prophet['yhat'])
                df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

            # XGBoost prediction
            xgb_series = None
            if (scaler is not None) and (feature_names is not None) and (xgboost_model is not None):
                X = df_proc.drop('PriceMarket', axis=1, errors='ignore').reindex(columns=feature_names, fill_value=0)
                try:
                    X_scaled = scaler.transform(X)
                except Exception:
                    X_scaled = X.values
                y_pred_log = xgboost_model.predict(X_scaled)
                y_pred = np.expm1(y_pred_log)
                xgb_series = pd.Series(y_pred, index=df_proc.index).rename('xgb')

            # Actuals
            actual_series = None
            if 'PriceMarket' in df_proc.columns:
                actual_series = df_proc['PriceMarket']
                if actual_series.max() < 10 and actual_series.mean() < 10:
                    actual_series = np.expm1(actual_series)

            # Build plotting dataframe
            if df_prophet is not None:
                df_plot = pd.DataFrame({'ds': df_prophet['ds']})
                df_plot['prophet'] = df_prophet.get('prophet')
                if xgb_series is not None:
                    df_plot = df_plot.merge(xgb_series.rename_axis('ds').reset_index(), on='ds', how='left')
                if actual_series is not None:
                    df_plot = df_plot.merge(actual_series.rename('actual').rename_axis('ds').reset_index(), on='ds', how='left')
            else:
                df_plot = pd.DataFrame({'ds': df_proc.index})
                if xgb_series is not None:
                    df_plot['xgb'] = xgb_series.values
                if actual_series is not None:
                    df_plot['actual'] = actual_series.values

            # Hybrid average
            if 'prophet' in df_plot.columns and 'xgb' in df_plot.columns:
                df_plot['hybrid'] = df_plot[['prophet','xgb']].mean(axis=1)
            elif 'prophet' in df_plot.columns:
                df_plot['hybrid'] = df_plot['prophet']
            elif 'xgb' in df_plot.columns:
                df_plot['hybrid'] = df_plot['xgb']

            # Plot
            hybrid_fig = plot_hybrid_chart(df_plot)
            st.pyplot(hybrid_fig)

            # Download
            st.download_button("⬇ Download Hybrid CSV", data=df_plot.to_csv(index=False).encode('utf-8'),
                               file_name="hybrid_forecast.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error building hybrid view: {e}")

    if st.button("🧾 Export PDF (Hybrid)"):
        kpis = [("MAE", f"{FIXED_MAE:.4f}"), ("RMSE", f"{FIXED_RMSE:.4f}"),
                ("MAPE", f"{FIXED_MAPE:.2f}%"), ("Accuracy", f"{FIXED_ACC:.2f}%")]
        figs = []
        if hybrid_fig is not None: figs.append(hybrid_fig)
        out = "/mnt/data/LayerX_Report_Hybrid.pdf"
        path = export_pdf(kpis, figs, out)
        st.success(f"PDF saved: {path}")
        st.markdown(f"[📥 Download PDF]({path})")
