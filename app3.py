import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

yf.pdr_override()

# Load the TensorFlow RNN model
model = tf.keras.models.load_model('keras_model.h5')

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
    st.title("JKSE Price Prediction using Recurrent Neural Network")

    ticker = st.text_input("Enter Stock Ticker", "^JKSE")
    n_days = st.slider("Number of days to forecast", min_value=1, max_value=30, value=10)
    end_date = st.date_input("Select End Date", value=pd.to_datetime("2024-04-17"))

    if st.button("Predict"):
        start = "2004-01-01"
        df = pdr.get_data_yahoo(ticker, start, end_date)
        final_df = pd.DataFrame(df['Close'])
        scaler = MinMaxScaler(feature_range=(0, 1))
        input_data = scaler.fit_transform(final_df)

        last_known_points = input_data[-100:]
        predictions, lower_bounds, upper_bounds = multistep_forecast_with_confidence(model, scaler, last_known_points, n_days)
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

if __name__ == "__main__":
    main()
