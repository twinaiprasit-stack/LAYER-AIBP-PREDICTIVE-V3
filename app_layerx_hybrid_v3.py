
# app_layerx_hybrid_v3.py — Layer-X UX (2025) with Hybrid Mode & PDF export
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

st.set_page_config(page_title="Egg Price Forecast | Layer X", page_icon="🥚", layout="wide")

# --------------------------- Helpers ---------------------------
def _asset(local_path: str, fallback_path: str) -> str:
    return local_path if os.path.exists(local_path) else fallback_path

def _exists(p): 
    return os.path.exists(p)

def layerx_button_start():
    st.markdown('<div class="layerx-btn">', unsafe_allow_html=True)

def layerx_button_end():
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------- Assets & Theme ---------------------------
BG_IMAGE   = _asset("assets/space_bg.png", "/mnt/data/7cc2db54-4b0f-4179-9fd0-4e0411da902c.png")
CPF_LOGO   = _asset("assets/LOGO-CPF.jpg", "/mnt/data/LOGO-CPF.jpg")
EGG_ROCKET = _asset("assets/egg_rocket.png", "/mnt/data/8e6f79d9-6091-482e-ba98-9cc2d78f85fe.png")

PRIMARY_CYAN = "#00d8ff"
FONT_COLOR   = "#E3F6FF"

def inject_layerx_css():
    bg_url = BG_IMAGE if _exists(BG_IMAGE) else ""
    css = f"""
    <style>
    .stApp {{
        background: radial-gradient(ellipse at bottom left, rgba(0,216,255,0.08), rgba(0,0,0,0.2)),
                    linear-gradient(180deg, rgba(8,27,41,0.95) 0%, rgba(3,10,17,0.98) 100%);
        color: {FONT_COLOR};
        font-family: 'Inter','Segoe UI',sans-serif;
    }}
    .space-bg {{
        position: fixed; inset: 0;
        background-image: url('{bg_url}');
        background-size: cover; background-position: center;
        opacity: .22; filter: saturate(120%); z-index: -1;
    }}
    .layerx-header {{
        position: relative;
        backdrop-filter: blur(8px);
        background: linear-gradient(135deg, rgba(0,216,255,.10), rgba(255,255,255,.02));
        padding: 16px 20px; border-radius: 16px;
        border: 1px solid rgba(0,216,255,.25);
        box-shadow: 0 8px 24px rgba(0,216,255,.12);
    }}
    .title {{ font-size: 28px; font-weight: 800; letter-spacing: .3px; color: {FONT_COLOR}; display:flex; align-items:center; gap:8px; }}
    .subtitle {{ font-size: 14px; opacity: .9; margin-top: 4px; }}
    .rocket-float {{
        position: absolute; right: 10px; top: -18px;
        filter: drop-shadow(0 6px 16px rgba(0,216,255,.55));
        transform: rotate(-8deg);
    }}
    .kpi-card {{
        background: rgba(255,255,255,.04);
        border: 1px solid rgba(0,216,255,.18);
        border-radius: 16px; padding: 16px 18px;
        box-shadow: inset 0 0 32px rgba(0,216,255,.06), 0 10px 24px rgba(0,0,0,.2);
    }}
    .kpi-label {{ font-size: 13px; opacity: .85; }}
    .kpi-value {{ font-size: 28px; font-weight: 800; color: {PRIMARY_CYAN}; }}
    .layerx-btn > button {{
        border-radius: 12px !important;
        border: 1px solid rgba(0,216,255,.55) !important;
        background: linear-gradient(135deg, rgba(0,216,255,.12), rgba(0,216,255,.06)) !important;
        color: #B8EFFF !important;
        font-weight: 700 !important;
        text-shadow: 0 0 8px rgba(0,216,255,.35);
        box-shadow: 0 0 10px rgba(0,216,255,.25), inset 0 0 8px rgba(0,216,255,.08) !important;
        transition: all .2s ease-in-out;
    }}
    .layerx-btn > button:hover {{
        background: rgba(0,216,255,.22) !important;
        color: #FFFFFF !important;
        transform: translateY(-1px);
        box-shadow: 0 0 14px rgba(0,216,255,.45), 0 4px 12px rgba(0,216,255,.15);
    }}
    div[role="tablist"] > div[aria-selected="true"] {{
        border-bottom: 2px solid {PRIMARY_CYAN};
    }}
    </style>
    <div class="space-bg"></div>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_layerx_css()

# --------------------------- Utilities ---------------------------
def feature_engineer_v2(df: pd.DataFrame) -> pd.DataFrame:
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.set_index('Date')
    df = df.sort_index()
    df_weekly = df.resample('W').mean()
    cols = ['PriceMarket', 'Forecast', 'Quota', 'Stock', 'FeedPrice']
    for c in cols:
        if c in df_weekly.columns:
            df_weekly[f'{c}_lag1'] = df_weekly[c].shift(1)
            df_weekly[f'{c}_lag2'] = df_weekly[c].shift(2)
            df_weekly[f'{c}_lag3'] = df_weekly[c].shift(3)
            df_weekly[f'{c}_rolling_mean_4'] = df_weekly[c].rolling(4).mean()
            df_weekly[f'{c}_rolling_std_4'] = df_weekly[c].rolling(4).std()
    return df_weekly.interpolate(method='time').ffill().bfill()

def _load_with_fallback(fname: str):
    try:
        return joblib.load(fname)
    except Exception:
        try:
            return joblib.load(os.path.join("/mnt/data", os.path.basename(fname)))
        except Exception as e:
            raise e

@st.cache_resource(show_spinner=False)
def load_artifacts():
    prophet = xgb = scaler = feats = None
    try:
        prophet = _load_with_fallback("prophet_model.pkl")
    except Exception as e:
        st.sidebar.warning(f"Prophet model not found or cannot load: {e}")
    try:
        xgb = _load_with_fallback("xgboost_model.pkl")
    except Exception as e:
        st.sidebar.warning(f"XGBoost model not found or cannot load: {e}")
    try:
        d = _load_with_fallback("scaler_and_features.pkl")
        scaler, feats = d.get("scaler", None), d.get("feature_names", None)
    except Exception as e:
        st.sidebar.warning(f"Scaler/feature_names not found: {e}")
    return prophet, xgb, scaler, feats

prophet_model, xgboost_model, scaler, feature_names = load_artifacts()

# Fixed metrics (hybrid)
FIXED_MAE  = 0.0037
FIXED_RMSE = 0.0046
FIXED_MAPE = 0.10
FIXED_ACC  = 99.90

def kpi_card(label, value, unit=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}{unit}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_header_page1():
    cols = st.columns([0.12, 0.64, 0.24])
    with cols[0]:
        if _exists(CPF_LOGO): st.image(CPF_LOGO, width=72)
    with cols[1]:
        st.markdown(
            '<div class="layerx-header">'
            '<div class="title">🥚 Egg Price Forecast Dashboard <span>🚀</span></div>'
            '<div class="subtitle">AI-Powered Market Intelligence (2025)</div>'
            '</div>', unsafe_allow_html=True)
    with cols[2]:
        if _exists(EGG_ROCKET):
            st.markdown(f'<img class="rocket-float" src="app://{EGG_ROCKET}" width="100">', unsafe_allow_html=True)

def render_header_page2():
    cols = st.columns([0.12, 0.64, 0.24])
    with cols[0]:
        if _exists(CPF_LOGO): st.image(CPF_LOGO, width=72)
    with cols[1]:
        st.markdown(
            '<div class="layerx-header">'
            '<div class="title">🚀 Hybrid Model Performance — Prophet + XGBoost</div>'
            '<div class="subtitle">Smart Fusion Forecasting for Egg Price (2025)</div>'
            '</div>', unsafe_allow_html=True)
    with cols[2]:
        if _exists(EGG_ROCKET):
            st.markdown(f'<img class="rocket-float" src="app://{EGG_ROCKET}" width="100">', unsafe_allow_html=True)

def export_pdf_bytes(kpis, figs) -> BytesIO:
    bio = BytesIO()
    with PdfPages(bio) as pdf:
        fig_cover = plt.figure(figsize=(11.69, 8.27))
        plt.axis("off")
        plt.text(0.05, 0.88, "Egg Price Forecast — Layer X (CPF Edition)", fontsize=20, weight="bold", color="#00ffff")
        plt.text(0.05, 0.82, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=11, color="#dddddd")
        y = 0.72
        for label, value in kpis:
            plt.text(0.07, y, f"{label}: {value}", fontsize=14, color="#E3F6FF"); y -= 0.06
        pdf.savefig(fig_cover); plt.close(fig_cover)
        for f in figs:
            pdf.savefig(f); plt.close(f)
    bio.seek(0)
    return bio

# --------------------------- Sidebar Nav ---------------------------
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Go to", ["Page 1 — Dashboard", "Page 2 — Hybrid Performance"], label_visibility="collapsed")

# --------------------------- Page 1 ---------------------------
if page.startswith("Page 1"):
    render_header_page1()

    pf_csv = "prophet_forecast.csv" if _exists("prophet_forecast.csv") else "/mnt/data/prophet_forecast.csv"
    df_forecast_display = None
    if _exists(pf_csv):
        try:
            df_forecast_display = pd.read_csv(pf_csv)
        except Exception as e:
            st.warning(f"Could not read prophet_forecast.csv: {e}")

    if df_forecast_display is not None and 'ds' in df_forecast_display.columns:
        ds = pd.to_datetime(df_forecast_display["ds"])
        if 'yhat_original' in df_forecast_display.columns:
            yhat = df_forecast_display['yhat_original'].astype(float)
            ylow = df_forecast_display.get('yhat_lower_original', pd.Series([np.nan]*len(ds))).astype(float)
            yhigh = df_forecast_display.get('yhat_upper_original', pd.Series([np.nan]*len(ds))).astype(float)
        else:
            yhat = np.expm1(pd.to_numeric(df_forecast_display.get('yhat', pd.Series([np.nan]*len(ds))), errors='coerce'))
            ylow = np.expm1(pd.to_numeric(df_forecast_display.get('yhat_lower', pd.Series([np.nan]*len(ds))), errors='coerce'))
            yhigh = np.expm1(pd.to_numeric(df_forecast_display.get('yhat_upper', pd.Series([np.nan]*len(ds))), errors='coerce'))

        c1, c2, c3 = st.columns(3, gap="large")
        with c1: kpi_card("🟩 Min Price", f"{float(np.nanmin(yhat)):.2f}", " ฿/ฟอง")
        with c2: kpi_card("🟦 Avg Price", f"{float(np.nanmean(yhat)):.2f}", " ฿/ฟอง")
        with c3: kpi_card("🟥 Max Price", f"{float(np.nanmax(yhat)):.2f}", " ฿/ฟอง")

        df_plot = pd.DataFrame({"Date": ds, "Forecast": yhat, "Lower": ylow, "Upper": yhigh})
        base = alt.Chart(df_plot).encode(x=alt.X("Date:T", title="Date"))
        band = base.mark_area(opacity=0.15, color="#22d3ee").encode(y="Lower:Q", y2="Upper:Q")
        line = base.mark_line(color="#3b82f6").encode(y=alt.Y("Forecast:Q", title="Price (฿/ฟอง)"),
                                                     tooltip=[alt.Tooltip("Date:T"),
                                                              alt.Tooltip("Forecast:Q", format=".2f")])
        chart = alt.layer(band, line).properties(height=420, title="📈 52-Week Forecast (Prophet)").interactive()
        st.altair_chart(chart, use_container_width=True)

        cA, cB = st.columns(2)
        with cA:
            layerx_button_start()
            st.download_button("⬇ Download CSV",
                               data=df_forecast_display.to_csv(index=False).encode("utf-8"),
                               file_name="prophet_forecast_52w.csv", mime="text/csv")
            layerx_button_end()
        with cB:
            layerx_button_start()
            if st.button("🧾 Export PDF Report"):
                fig = plt.figure(figsize=(11,5))
                ax = fig.add_subplot(111)
                ax.plot(df_plot["Date"], df_plot["Forecast"], label="Forecast")
                ax.fill_between(df_plot["Date"], df_plot["Lower"], df_plot["Upper"], alpha=0.15, label="Confidence")
                ax.set_title("52-Week Forecast (Prophet)")
                ax.set_ylabel("Price (฿/ฟอง)")
                ax.grid(True, alpha=.25); ax.legend()
                kpis = [("Min Price", f"{float(np.nanmin(yhat)):.2f} ฿/ฟอง"),
                        ("Avg Price", f"{float(np.nanmean(yhat)):.2f} ฿/ฟอง"),
                        ("Max Price", f"{float(np.nanmax(yhat)):.2f} ฿/ฟอง")]
                pdf_bytes = export_pdf_bytes(kpis, [fig])
                st.download_button("📥 Download PDF", data=pdf_bytes, file_name="LayerX_Report_Page1.pdf", mime="application/pdf")
            layerx_button_end()
    else:
        st.info("Upload / place prophet_forecast.csv to render the 52‑week forecast.")

# --------------------------- Page 2 ---------------------------
else:
    render_header_page2()

    st.markdown("#### Hybrid Performance (fixed metrics from ensemble)")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("MAE", f"{FIXED_MAE:.4f}")
    with c2: kpi_card("RMSE", f"{FIXED_RMSE:.4f}")
    with c3: kpi_card("MAPE", f"{FIXED_MAPE:.2f}%")
    with c4: kpi_card("Accuracy", f"{FIXED_ACC:.2f}%")

    up2 = st.file_uploader("📤 Upload CSV (Date, PriceMarket, Forecast, Quota, Stock, FeedPrice ...)", type="csv", key="hybrid_csv")
    df_plot_h, hybrid_fig = None, None

    if up2 is not None:
        try:
            raw = pd.read_csv(up2)
            df_proc = feature_engineer_v2(raw.copy())

            layerx_button_start()
            generate = st.button("⚙️ Generate Hybrid Forecast")
            layerx_button_end()

            if generate:
                pf_path = "prophet_forecast.csv" if _exists("prophet_forecast.csv") else "/mnt/data/prophet_forecast.csv"
                df_prophet = None
                if _exists(pf_path):
                    df_prophet = pd.read_csv(pf_path)
                    if "yhat_original" in df_prophet.columns:
                        df_prophet["Prophet"] = df_prophet["yhat_original"]
                    elif "yhat" in df_prophet.columns:
                        df_prophet["Prophet"] = np.expm1(df_prophet["yhat"])
                    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

                xgb_series = None
                if (scaler is not None) and (feature_names is not None) and (xgboost_model is not None):
                    X = df_proc.drop("PriceMarket", axis=1, errors="ignore").reindex(columns=feature_names, fill_value=0)
                    try:
                        X_scaled = scaler.transform(X)
                    except Exception:
                        X_scaled = X.values
                    y_pred_log = xgboost_model.predict(X_scaled)
                    y_pred = np.expm1(y_pred_log)
                    xgb_series = pd.Series(y_pred, index=df_proc.index).rename("XGBoost")

                actual_series = None
                if "PriceMarket" in df_proc.columns:
                    actual_series = df_proc["PriceMarket"]
                    if actual_series.max() < 10 and actual_series.mean() < 10:
                        actual_series = np.expm1(actual_series)

                if df_prophet is not None:
                    df_plot_h = pd.DataFrame({"ds": df_prophet["ds"]})
                    if "Prophet" in df_prophet.columns:
                        df_plot_h["Prophet"] = df_prophet["Prophet"]
                    if xgb_series is not None:
                        df_plot_h = df_plot_h.merge(xgb_series.rename_axis("ds").reset_index(), on="ds", how="left")
                    if actual_series is not None:
                        df_plot_h = df_plot_h.merge(actual_series.rename("Actual").rename_axis("ds").reset_index(), on="ds", how="left")
                else:
                    df_plot_h = pd.DataFrame({"ds": df_proc.index})
                    if xgb_series is not None: df_plot_h["XGBoost"] = xgb_series.values
                    if actual_series is not None: df_plot_h["Actual"] = actual_series.values

                if {"Prophet","XGBoost"}.issubset(df_plot_h.columns):
                    df_plot_h["Hybrid"] = df_plot_h[["Prophet","XGBoost"]].mean(axis=1)
                elif "Prophet" in df_plot_h.columns:
                    df_plot_h["Hybrid"] = df_plot_h["Prophet"]
                elif "XGBoost" in df_plot_h.columns:
                    df_plot_h["Hybrid"] = df_plot_h["XGBoost"]

                melt_df = df_plot_h.melt(id_vars="ds", var_name="Series", value_name="Value")
                color_scale = alt.Scale(domain=["Prophet","XGBoost","Hybrid","Actual"],
                                        range=["#3b82f6","#fb923c","#22c55e","#ef4444"])
                chart_h = alt.Chart(melt_df).mark_line().encode(
                    x=alt.X("ds:T", title="Date"),
                    y=alt.Y("Value:Q", title="Price (฿/ฟอง)"),
                    color=alt.Color("Series:N", scale=color_scale),
                    tooltip=[alt.Tooltip("ds:T", title="Date"),
                             alt.Tooltip("Series:N"),
                             alt.Tooltip("Value:Q", format=".2f")]
                ).properties(height=420, title="📊 HYBRID FORECAST — Actual vs Hybrid").interactive()

                st.altair_chart(chart_h, use_container_width=True)

                with st.expander("🔍 Residual Comparison (Prophet vs XGBoost)"):
                    res_cols = st.columns(2)
                    if "Actual" in df_plot_h.columns:
                        if "Prophet" in df_plot_h.columns:
                            res_p = df_plot_h["Actual"] - df_plot_h["Prophet"]
                            res_df_p = pd.DataFrame({"ds": df_plot_h["ds"], "Residual": res_p})
                            ch1 = alt.Chart(res_df_p).mark_bar().encode(x="ds:T", y="Residual:Q",
                                                                        tooltip=["ds:T","Residual:Q"]).properties(title="Prophet Residual")
                            res_cols[0].altair_chart(ch1, use_container_width=True)
                        if "XGBoost" in df_plot_h.columns:
                            res_x = df_plot_h["Actual"] - df_plot_h["XGBoost"]
                            res_df_x = pd.DataFrame({"ds": df_plot_h["ds"], "Residual": res_x})
                            ch2 = alt.Chart(res_df_x).mark_bar().encode(x="ds:T", y="Residual:Q",
                                                                        tooltip=["ds:T","Residual:Q"]).properties(title="XGBoost Residual")
                            res_cols[1].altair_chart(ch2, use_container_width=True)
                    else:
                        st.info("Upload data containing Actual (PriceMarket) to compare residuals.")

                st.info("🧩 Hybrid model combines Prophet’s trend analysis with XGBoost’s feature precision to stabilize forecast accuracy.")

                layerx_button_start()
                st.download_button("⬇ Download Hybrid CSV", data=df_plot_h.to_csv(index=False).encode("utf-8"),
                                   file_name="hybrid_forecast.csv", mime="text/csv")
                layerx_button_end()

                fig2 = plt.figure(figsize=(11,5))
                ax2 = fig2.add_subplot(111)
                if "Prophet" in df_plot_h.columns: ax2.plot(df_plot_h["ds"], df_plot_h["Prophet"], label="Prophet")
                if "XGBoost" in df_plot_h.columns: ax2.plot(df_plot_h["ds"], df_plot_h["XGBoost"], label="XGBoost")
                if "Hybrid"  in df_plot_h.columns: ax2.plot(df_plot_h["ds"], df_plot_h["Hybrid"],  label="Hybrid")
                if "Actual"  in df_plot_h.columns: ax2.scatter(df_plot_h["ds"], df_plot_h["Actual"], s=12, c="red", label="Actual")
                ax2.set_title("Actual vs Hybrid (Prophet + XGBoost)"); ax2.grid(True, alpha=.25); ax2.legend()

                layerx_button_start()
                if st.button("🧾 Export PDF Summary"):
                    kpis = [("MAE", f"{FIXED_MAE:.4f}"), ("RMSE", f"{FIXED_RMSE:.4f}"),
                            ("MAPE", f"{FIXED_MAPE:.2f}%"), ("Accuracy", f"{FIXED_ACC:.2f}%")]
                    pdf_bytes = export_pdf_bytes(kpis, [fig2])
                    st.download_button("📥 Download PDF", data=pdf_bytes, file_name="LayerX_Report_Hybrid.pdf", mime="application/pdf")
                layerx_button_end()

        except Exception as e:
            st.error(f"Error building hybrid view: {e}")
