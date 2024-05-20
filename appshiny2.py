from shiny import App, render, ui
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
        pred_rescaled = scaler.inverse_transform([[pred*1.05]])[0, 0]
        predictions.append(pred_rescaled)

        error = pred_rescaled * confidence_pct
        lower_bounds.append(pred_rescaled - error)
        upper_bounds.append(pred_rescaled + error)

        new_forecast = np.append(forecast[0, 1:, 0], pred*1.05).reshape(1, 100, 1)
        forecast = new_forecast

    return predictions, lower_bounds, upper_bounds

app_ui = ui.page_fluid(
    ui.panel_title("JKSE Price Prediction using Recurrent Neural Network"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_text("ticker", "Enter Stock Ticker", "^JKSE"),
            ui.input_slider("n_days", "Number of days to forecast", min=1, max=30, value=10),
            ui.input_date("end_date", "Select End Date", value="2024-04-17", min="2000-01-01", max="2024-12-31"),
            ui.input_action_button("predict", "Predict")
        ),
        ui.panel_main(
            ui.output_table("price_table"),
            ui.output_plot("price_plot")
        )
    )
)

def server(input, output, session):
    @output
    @render.table
    def price_table():
        ticker = input.ticker()
        n_days = input.n_days()
        end_date = input.end_date()
        if not ticker:
            return "Please enter a stock ticker."

        start = "2004-01-01"
        df = pdr.get_data_yahoo(ticker, start, end_date)
        final_df = pd.DataFrame(df['Close'])
        scaler = MinMaxScaler(feature_range=(0, 1))
        input_data = scaler.fit_transform(final_df)

        last_known_points = input_data[-100:]
        predictions, lower_bounds, upper_bounds = multistep_forecast_with_confidence(model, scaler, last_known_points, n_days)
        future_dates = pd.date_range(end_date, periods=n_days+1, freq='B')[1:]  # Generate future dates

        # Create a column that numbers the forecast days from 1 to n_days
        forecast_days = range(1, n_days + 1)

        predictions_df = pd.DataFrame({
            'Forecast n Days Ahead': forecast_days,
            'Predicted Close': predictions,
            'Lower Bound': lower_bounds,
            'Upper Bound': upper_bounds
        }, index=future_dates)
        predictions_df.index.name = 'Date'  # Set 'Date' as the index for clarity

        return predictions_df


    @output
    @render.plot
    def price_plot():
        ticker = input.ticker()
        n_days = input.n_days()
        end_date = input.end_date()
        if not ticker:
            return "Please enter a stock ticker."

        start = "2004-01-01"
        df = pdr.get_data_yahoo(ticker, start, end_date)
        final_df = pd.DataFrame(df['Close'])
        scaler = MinMaxScaler(feature_range=(0, 1))
        input_data = scaler.fit_transform(final_df)

        last_known_points = input_data[-100:]
        predictions, lower_bounds, upper_bounds = multistep_forecast_with_confidence(model, scaler, last_known_points, n_days)
        future_dates = pd.date_range(end_date, periods=n_days+1, freq='B')[1:]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index[-100:], df['Close'][-100:], 'g', label="Original Price")
        ax.plot(future_dates, predictions, 'r', label="Forecasted Price")
        ax.fill_between(future_dates, lower_bounds, upper_bounds, color='orange', alpha=0.3, label="Confidence Interval")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.set_title('Historical and Forecasted Closing Prices')
        ax.legend()
        return fig

app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
