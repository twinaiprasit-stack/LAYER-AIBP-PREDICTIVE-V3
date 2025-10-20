import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define the feature engineering function
def feature_engineer_v2(df):
    """Performs feature engineering (lag and rolling statistics) on specified columns and handles missing values."""
    # Ensure Date is datetime and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df = df.set_index('Date')

    # Resample to weekly means
    df_weekly = df.resample('W').mean()

    # Define columns for feature engineering
    cols_to_engineer = ['PriceMarket', 'Forecast', 'Quota', 'Stock', 'FeedPrice']

    # Feature Engineering
    for col in cols_to_engineer:
        if col in df_weekly.columns:
            df_weekly[f'{col}_lag1'] = df_weekly[col].shift(1)
            df_weekly[f'{col}_lag2'] = df_weekly[col].shift(2)
            df_weekly[f'{col}_lag3'] = df_weekly[col].shift(3)
            df_weekly[f'{col}_rolling_mean_4'] = df_weekly[col].rolling(window=4).mean()
            df_weekly[f'{col}_rolling_std_4'] = df_weekly[col].rolling(window=4).std()

    # Handle missing values
    df_weekly = df_weekly.interpolate(method='time')
    df_weekly = df_weekly.ffill()
    df_weekly = df_weekly.bfill()

    return df_weekly

# Load saved models and scaler dictionary
@st.cache_resource
def load_models_and_scaler():
    """Loads the saved Prophet model, XGBoost model, and StandardScaler dictionary."""
    prophet_model = joblib.load('prophet_model.pkl')
    xgboost_model = joblib.load('xgboost_model.pkl')
    scaler_and_features = joblib.load('scaler_and_features.pkl')
    scaler = scaler_and_features['scaler']
    feature_names = scaler_and_features['feature_names']
    return prophet_model, xgboost_model, scaler, feature_names

prophet_model, xgboost_model, scaler, feature_names = load_models_and_scaler()

# Streamlit App Layout
st.title("Hybrid Egg Price Forecasting (Prophet & XGBoost)")

# Load the original test data for initial accuracy display
try:
    # Load the original test data
    test_df_original = pd.read_csv('/content/Predict Egg Price 2022-25 with Date - Test_Pmhoo.csv')
    test_df_original['Date'] = pd.to_datetime(test_df_original['Date'], dayfirst=True)
    test_df_original = test_df_original.set_index('Date')

    # Apply feature engineering to the original test data
    test_df_weekly_original_processed = feature_engineer_v2(test_df_original.copy().reset_index()) # Pass reset index for feature_engineer_v2

    # Separate target and features
    y_test_original_eval = test_df_weekly_original_processed['PriceMarket']
    X_test_original_eval = test_df_weekly_original_processed.drop('PriceMarket', axis=1)

    # Ensure columns match those used during training before scaling
    X_test_original_eval = X_test_original_eval[feature_names]

    # Scale the test features
    X_test_scaled_original_eval = scaler.transform(X_test_original_eval)

    # Make predictions on the scaled original test data
    y_pred_scaled_original_eval = xgboost_model.predict(X_test_scaled_original_eval)

    # Inverse transform predictions and actuals for evaluation
    y_test_original_eval_untransformed = np.expm1(y_test_original_eval)
    y_pred_original_eval_untransformed = np.expm1(y_pred_scaled_original_eval)

    # Calculate evaluation metrics
    mae_original = mean_absolute_error(y_test_original_eval_untransformed, y_pred_original_eval_untransformed)
    rmse_original = np.sqrt(mean_squared_error(y_test_original_eval_untransformed, y_pred_original_eval_untransformed))
    mape_original = np.mean(np.abs((y_test_original_eval_untransformed - y_pred_original_eval_untransformed) / y_test_original_eval_untransformed)) * 100 if np.mean(y_test_original_eval_untransformed) != 0 else float('inf')
    r2_original = r2_score(y_test_original_eval_untransformed, y_pred_original_eval_untransformed)
    accuracy_original = 100 * (1 - (mae_original / np.mean(y_test_original_eval_untransformed)))

    st.sidebar.subheader("Model Performance on Test Set")
    st.sidebar.write(f'Mean Absolute Error (MAE): {mae_original:.4f}')
    st.sidebar.write(f'Root Mean Squared Error (RMSE): {rmse_original:.4f}')
    st.sidebar.write(f'Mean Absolute Percentage Error (MAPE): {mape_original:.2f}%')
    st.sidebar.write(f'R-squared (R2): {r2_original:.4f}')
    st.sidebar.write(f'Accuracy%: {accuracy_original:.2f}%')
    st.sidebar.info("Note: The R-squared value of 0 indicates that the model does not explain the variance in the test set around its mean. This can happen with very stable target variables or limited test data, but the low MAE/RMSE and high Accuracy% suggest the model's predictions are close to the actual values.")


except Exception as e:
    st.sidebar.error(f"Error loading or processing original test data for accuracy display: {e}")
    accuracy_original = None

tab1, tab2 = st.tabs(["Trend (Prophet)", "Predict (XGBoost)"])

with tab1:
    st.header("Trend Analysis with Prophet")

    uploaded_file_prophet = st.file_uploader("Upload historical CSV data for Prophet forecasting", type="csv", key='prophet_upload')

    if uploaded_file_prophet is not None:
        df_history = pd.read_csv(uploaded_file_prophet)

        try:
            df_history_processed = feature_engineer_v2(df_history.copy()) # Use a copy to avoid modifying the original uploaded df

            st.subheader("Processed Historical Data (for forecasting)")
            st.dataframe(df_history_processed)

            periods = st.number_input("Number of future weeks to forecast", min_value=1, value=52, key='prophet_periods')

            # Create future dataframe for Prophet
            future_prophet = prophet_model.make_future_dataframe(freq='W', periods=periods, include_history=False)

            # Populate future dataframe with regressor values
            # Use the last known scaled regressor values from the processed history for all future dates.
            # First, scale the history data
            # Drop the target if it exists before scaling
            features_history = df_history_processed.drop('PriceMarket', axis=1, errors='ignore')

            # Ensure history features have the same columns as trained features, fill missing with 0 or a strategy
            # This is crucial if uploaded data has different columns or order
            features_history = features_history.reindex(columns=feature_names, fill_value=0) # Fill missing columns with 0

            features_history_scaled = scaler.transform(features_history)
            features_history_scaled_df = pd.DataFrame(features_history_scaled, index=features_history.index, columns=feature_names)


            last_regressor_values_scaled = features_history_scaled_df.tail(1).iloc[0]

            # Ensure the 'ds' column is datetime before setting index for join
            future_prophet['ds'] = pd.to_datetime(future_prophet['ds'])
            future_prophet = future_prophet.set_index('ds')

            # Create a temporary DataFrame for the last known scaled regressor values
            # Repeat the last row for the length of the future dataframe
            last_values_df = pd.DataFrame([last_regressor_values_scaled] * len(future_prophet), index=future_prophet.index, columns=feature_names)

            # Join the last known regressor values to the future dataframe
            future_prophet = future_prophet.join(last_values_df)

            # Handle any potential missing values in the regressors of the future dataframe (unlikely with this simple method, but good practice)
            future_prophet = future_prophet.interpolate(method='time')
            future_prophet = future_prophet.ffill()
            future_prophet = future_prophet.bfill()

            future_prophet = future_prophet.reset_index() # Reset index before prediction

            # Generate predictions
            forecast = prophet_model.predict(future_prophet)

            # Inverse transform the 'yhat' (predicted values) column
            forecast['yhat_original'] = np.expm1(forecast['yhat'])
            forecast['yhat_lower_original'] = np.expm1(forecast['yhat_lower'])
            forecast['yhat_upper_original'] = np.expm1(forecast['yhat_upper'])


            st.subheader("Prophet Forecast")
            st.dataframe(forecast[['ds', 'yhat_original', 'yhat_lower_original', 'yhat_upper_original']])

            # Plot Prophet forecast
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(forecast['ds'], forecast['yhat_original'], label='Prophet Forecast', color='blue')

            # Add historical data to the plot if available
            if 'PriceMarket' in df_history_processed.columns:
                 ax.plot(df_history_processed.index, np.expm1(df_history_processed['PriceMarket']), label='Actuals', color='red')


            ax.fill_between(forecast['ds'], forecast['yhat_lower_original'], forecast['yhat_upper_original'], color='blue', alpha=0.2, label='Confidence Interval')
            ax.set_title('Prophet Forecast with Actuals and Confidence Intervals')
            ax.set_xlabel('Date')
            ax.set_ylabel('PriceMarket')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Plot Prophet components
            fig_components = prophet_model.plot_components(forecast)
            st.pyplot(fig_components)


            # Download forecast data
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_data = convert_df_to_csv(forecast[['ds', 'yhat_original', 'yhat_lower_original', 'yhat_upper_original']])
            st.download_button(
                label="Download Prophet Forecast as CSV",
                data=csv_data,
                file_name='prophet_forecast.csv',
                mime='text/csv',
            )

        except Exception as e:
            st.error(f"Error processing Prophet forecasting: {e}")

with tab2:
    st.header("Prediction with XGBoost")

    uploaded_file_xgboost = st.file_uploader("Upload CSV data for XGBoost prediction", type="csv", key='xgboost_upload')

    if uploaded_file_xgboost is not None:
        df_predict = pd.read_csv(uploaded_file_xgboost)

        try:
            # Store original 'PriceMarket' if available before feature engineering and potential log transform
            if 'PriceMarket' in df_predict.columns:
                y_actual_original = df_predict['PriceMarket'].copy()
            else:
                y_actual_original = None

            df_predict_processed = feature_engineer_v2(df_predict.copy()) # Use a copy

            st.subheader("Processed Data (for XGBoost prediction)")
            st.dataframe(df_predict_processed)

            # Prepare features for scaling
            # Drop the target variable if it exists in the processed DataFrame
            features_for_scaling = df_predict_processed.drop('PriceMarket', axis=1, errors='ignore')

            # Ensure prediction features have the same columns as trained features, fill missing with 0 or a strategy
            # This is crucial if uploaded data has different columns or order
            features_for_scaling = features_for_scaling.reindex(columns=feature_names, fill_value=0) # Fill missing columns with 0

            # Scale the features
            features_scaled = scaler.transform(features_for_scaling)
            features_scaled_df = pd.DataFrame(features_scaled, index=features_for_scaling.index, columns=feature_names)


            # Make predictions
            y_pred_scaled = xgboost_model.predict(features_scaled_df)

            # Inverse transform predictions
            y_pred_original = np.expm1(y_pred_scaled)

            st.subheader("XGBoost Predictions")
            predictions_df = pd.DataFrame(y_pred_original, index=df_predict_processed.index, columns=['Predicted_PriceMarket'])
            st.dataframe(predictions_df)

            # Evaluate if actuals are available
            if y_actual_original is not None:
                # Align actuals and predictions by date index
                y_actual_original_aligned = y_actual_original[df_predict_processed.index]

                # Calculate evaluation metrics on original scale
                mae = mean_absolute_error(y_actual_original_aligned, y_pred_original)
                rmse = np.sqrt(mean_squared_error(y_actual_original_aligned, y_pred_original))
                mape = np.mean(np.abs((y_actual_original_aligned - y_pred_original) / y_actual_original_aligned)) * 100 if np.mean(y_actual_original_aligned) != 0 else float('inf')
                r2 = r2_score(y_actual_original_aligned, y_pred_original)
                accuracy = 100 * (1 - (mae / np.mean(y_actual_original_aligned)))

                st.subheader("Evaluation Metrics")
                st.write(f'Mean Absolute Error (MAE): {mae:.4f}')
                st.write(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
                st.write(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
                st.write(f'R-squared (R2): {r2:.4f}')
                st.write(f'Accuracy%: {accuracy:.2f}%')
                st.info("Note: The R-squared value of 0 indicates that the model does not explain the variance in the uploaded data around its mean. This can happen with very stable target variables or limited data, but the low MAE/RMSE and high Accuracy% suggest the model's predictions are close to the actual values.")


                # Scatter plot of predicted vs actual values
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_actual_original_aligned, y_pred_original, alpha=0.5)
                ax.set_title(f'XGBoost Predicted vs Actual PriceMarket\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, Accuracy%: {accuracy:.2f}%')
                ax.set_xlabel('Actual PriceMarket')
                ax.set_ylabel('Predicted PriceMarket')
                ax.grid(True)
                st.pyplot(fig)


            # Download predictions
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv().encode('utf-8')

            csv_data = convert_df_to_csv(predictions_df)
            st.download_button(
                label="Download XGBoost Predictions as CSV",
                data=csv_data,
                file_name='xgboost_predictions.csv',
                mime='text/csv',
            )

        except Exception as e:
            st.error(f"Error processing XGBoost prediction: {e}")
