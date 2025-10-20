# ส่วนประกาศ assets
BG_IMAGE   = _asset("assets/space_bg.png", "/mnt/data/7cc2db54-4b0f-4179-9fd0-4e0411da902c.png")
CPF_LOGO   = _asset("assets/LOGO-CPF.jpg", "/mnt/data/LOGO-CPF.jpg")
EGG_ROCKET = _asset("assets/egg_rocket.png", "/mnt/data/8e6f79d9-6091-482e-ba98-9cc2d78f85fe.png")

# CSS ปรับสีปุ่ม Export / Download ให้ชัดขึ้น
    /* Buttons */
    .layerx-btn > button {
        border-radius: 12px !important;
        border: 1px solid rgba(0,216,255,0.55) !important;
        background: linear-gradient(135deg, rgba(0,216,255,0.12), rgba(0,216,255,0.06)) !important;
        color: #B8EFFF !important;                  /* ✅ ตัวอักษรฟ้าสว่างขึ้น */
        font-weight: 600 !important;
        text-shadow: 0 0 8px rgba(0,216,255,0.35);
        box-shadow: 0 0 10px rgba(0,216,255,0.25), inset 0 0 8px rgba(0,216,255,0.08) !important;
        transition: all 0.2s ease-in-out;
    }
    .layerx-btn > button:hover {
        background: rgba(0,216,255,0.22) !important;
        color: #FFFFFF !important;                 /* ✅ ขาวเมื่อ hover */
        transform: translateY(-1px);
        box-shadow: 0 0 14px rgba(0,216,255,0.45), 0 4px 12px rgba(0,216,255,0.15);
    }

# ---------------- Hybrid Page ----------------

up2 = st.file_uploader("📤 Upload CSV (Date, PriceMarket, Forecast, Quota, Stock, FeedPrice ...)", type="csv", key="hybrid_csv")
df_plot, hybrid_fig = None, None

if up2 is not None:
    try:
        raw = pd.read_csv(up2)
        df_proc = feature_engineer_v2(raw.copy())

        # ปุ่ม Generate แยกจาก Upload
        if st.button("⚙️ Generate Hybrid Forecast"):
            pf_path = "prophet_forecast.csv" if os.path.exists("prophet_forecast.csv") else "/mnt/data/prophet_forecast.csv"
            df_prophet = None
            if os.path.exists(pf_path):
                df_prophet = pd.read_csv(pf_path)
                if 'yhat_original' in df_prophet.columns:
                    df_prophet['prophet'] = df_prophet['yhat_original']
                elif 'yhat' in df_prophet.columns:
                    df_prophet['prophet'] = np.expm1(df_prophet['yhat'])
                df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

            # XGBoost model
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

            actual_series = None
            if 'PriceMarket' in df_proc.columns:
                actual_series = df_proc['PriceMarket']
                if actual_series.max() < 10 and actual_series.mean() < 10:
                    actual_series = np.expm1(actual_series)

            if df_prophet is not None:
                df_plot = pd.DataFrame({'ds': df_prophet['ds']})
                df_plot['prophet'] = df_prophet.get('prophet')
                if xgb_series is not None:
                    df_plot = df_plot.merge(xgb_series.rename_axis('ds').reset_index(), on='ds', how='left')
                if actual_series is not None:
                    df_plot = df_plot.merge(actual_series.rename('actual').rename_axis('ds').reset_index(), on='ds', how='left')
            else:
                df_plot = pd.DataFrame({'ds': df_proc.index})
                if xgb_series is not None: df_plot['xgb'] = xgb_series.values
                if actual_series is not None: df_plot['actual'] = actual_series.values

            if 'prophet' in df_plot.columns and 'xgb' in df_plot.columns:
                df_plot['hybrid'] = df_plot[['prophet', 'xgb']].mean(axis=1)
            elif 'prophet' in df_plot.columns:
                df_plot['hybrid'] = df_plot['prophet']
            elif 'xgb' in df_plot.columns:
                df_plot['hybrid'] = df_plot['xgb']

            hybrid_fig = plot_hybrid_chart(df_plot)
            st.pyplot(hybrid_fig)
            st.download_button("⬇ Download Hybrid CSV", data=df_plot.to_csv(index=False).encode('utf-8'),
                               file_name="hybrid_forecast.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error building hybrid view: {e}")
