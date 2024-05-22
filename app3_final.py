import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

yf.pdr_override()

# Load TensorFlow RNN model and Keras model
rnn_model = tf.keras.models.load_model('keras_model.h5')
multivar_model = load_model('multivar_model.h5')

# Function to make predictions using the multivariate model
std_deviation = 0.05 
def make_multivar_prediction_with_confidence(input_data, confidence_pct=0.025):
    # Make prediction
    prediction = multivar_model.predict(input_data)

    # Calculate the margin of error for the confidence interval
    # Using the Z-value for 95% confidence which is approximately 1.96 for a two-tailed distribution
    z_score = 1.96  # Corresponds to 95% confidence
    margin_error = z_score * std_deviation

    # Calculate lower and upper bounds
    lower_bounds = prediction - margin_error
    upper_bounds = prediction + margin_error

    return prediction, lower_bounds, upper_bounds

# Function for univariate forecasting with confidence intervals
def multistep_forecast_with_confidence(model, scaler, input_seq, n_steps, confidence_pct=0.025):
    forecast = input_seq[-100:].reshape(1, 100, 1)
    predictions = []
    lower_bounds = []
    upper_bounds = []

    for _ in range(n_steps):
        pred = model.predict(forecast)[0, 0]
        pred_rescaled = scaler.inverse_transform([[pred * 1.05]])[0, 0]
        predictions.append(pred_rescaled)

        error = pred_rescaled * confidence_pct
        lower_bounds.append(pred_rescaled - error)
        upper_bounds.append(pred_rescaled + error)

        new_forecast = np.append(forecast[0, 1:, 0], pred * 1.05).reshape(1, 100, 1)
        forecast = new_forecast

    return predictions, lower_bounds, upper_bounds

def main():
    st.title("JCI Index Projection Dashboard")

    # Tabs for different models
    tab1, tab2 = st.tabs(["Univariate Model", "Multivariate Model"])

    with tab1:
        st.subheader("JCI Index Projection using Recurrent Neural Network")
        n_days = st.slider("Number of days to forecast", min_value=1, max_value=30, value=10)
        end_date = st.date_input("Select End Date", value=pd.to_datetime("2024-05-17"))

        if st.button("Predict"):
            start = "2004-01-01"
            df = pdr.get_data_yahoo("^JKSE", start, end_date)
            final_df = pd.DataFrame(df['Close'])
            scaler = MinMaxScaler(feature_range=(0, 1))
            input_data = scaler.fit_transform(final_df)

            last_known_points = input_data[-100:]
            predictions, lower_bounds, upper_bounds = multistep_forecast_with_confidence(rnn_model, scaler, last_known_points, n_days)
            future_dates = pd.date_range(end_date, periods=n_days+1, freq='B')[1:]  # Generate future dates

            predictions_df = pd.DataFrame({
                'Predicted Close': predictions,
                'Lower Bound': lower_bounds,
                'Upper Bound': upper_bounds
            }, index=future_dates)

            st.subheader("Predicted Prices")
            st.write(predictions_df)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df.index[-100:], df['Close'][-100:], 'g', label="Original Price")
            ax.plot(future_dates, predictions, 'r', label="Forecasted Price")
            ax.fill_between(future_dates, lower_bounds, upper_bounds, color='orange', alpha=0.3, label="Confidence Interval")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.set_title('Historical and Forecasted Closing Prices')
            ax.legend()
            st.pyplot(fig)

    with tab2:
        st.subheader("JCI Index Projection with Economic Indicators using Neural Network")
        # User inputs for the multivariate model
        input_data = {
            'dow_jones_industrial_index': st.number_input('Dow Jones Industrial Index', format="%.2f"),
            'emas_(xau/usd)': st.number_input('Gold (XAU/USD)', format="%.2f"),
            'minyak_sawit_(crude_palm_oil)': st.number_input('Crude Palm Oil', format="%.2f"),
            'nilai_tukar_rupiah_terhadap_usd_(usd/idr)': st.number_input('USD/IDR Exchange Rate', format="%.2f"),
            'suku_bunga_acuan_bank_sentral_us_fed_fund_rate': st.number_input('US Fed Fund Rate', format="%.2f"),
            'yield_sun_10y': st.number_input('10-Year Government Bond Yield', format="%.2f"),
            'credit_default_swap_sun_5y': st.number_input('5-Year Credit Default Swap', format="%.2f"),
            'dollar_index_(dxy)': st.number_input('Dollar Index (DXY)', format="%.2f"),
            'cadangan_devisa_indonesia_(milyar_usd)': st.number_input('Indonesian Foreign Exchange Reserves (Billion USD)', format="%.2f"),
            'trade_balance_indonesia_(juta_usd)': st.number_input('Indonesian Trade Balance (Million USD)', format="%.2f"),
            'inflasi_indonesia': st.number_input('Inflation in Indonesia', format="%.2f"),
            'inflasi_us': st.number_input('Inflation in US', format="%.2f"),
        }
        input_df = pd.DataFrame([input_data])
        if st.button('Predict Now'):
            prediction, lower_bounds, upper_bounds = make_multivar_prediction_with_confidence(input_df)
            st.write('Prediction:', prediction[0][0])
            st.write('Lower Bound:', lower_bounds[0][0])
            st.write('Upper Bound:', upper_bounds[0][0])

if __name__ == "__main__":
    main()

# streamlit run app3.py