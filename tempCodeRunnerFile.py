# import liabraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

start="2004-01-01"
end="2024-04-17"

st.title('Stock Closing Price Prediction')
# reading data
user_input = st.text_input('Enter Stock Ticker', '^JKSE')
df = pdr.get_data_yahoo(user_input, start, end)

st.subheader('Dated from 1st Jan, 2004 to 30th Apr, 2024')
st.write(df.describe())

# first plot
st.subheader('Closing Price Vs Time Chart')
fig1 = plt.figure(figsize = (12, 6))
plt.plot(df.Close)
st.pyplot(fig1)

# moving average
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

# second plot
st.subheader('Closing Price Vs Time Chart with 100 days Moving Average')
fig2 = plt.figure(figsize = (12, 6))
plt.plot(df.Close, 'r', label="Per Day Closing")
plt.plot(ma100, 'g', label="Moving Average 100")
st.pyplot(fig2)

# third plot
st.subheader('Closing Price Vs Time Chart with 100 days and 200 days Moving Average')
fig2 = plt.figure(figsize = (12, 6))
plt.plot(ma200, 'b', label="Moving Average 200")
plt.plot(ma100, 'g', label="Moving Average 100")
st.pyplot(fig2)

# Predict with model 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data as pdr

model = load_model('keras_model.h5')
def multistep_forecast_with_confidence(model, scaler, input_seq, n_steps, confidence_pct=0.05):
    # Reshape input to match the model's expected input shape
    forecast = input_seq[-100:].reshape(1, 100, 1)
    predictions = []
    lower_bounds = []
    upper_bounds = []

    for _ in range(n_steps):
        pred = model.predict(forecast)[0, 0]
        # Inverse transform the scaled prediction
        pred_rescaled = scaler.inverse_transform([[pred*1.05]])[0, 0]
        predictions.append(pred_rescaled)
        
        # Calculate confidence interval bounds
        lower_bound = pred_rescaled * (1 - confidence_pct)
        upper_bound = pred_rescaled * (1 + confidence_pct)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
        
        # Update the forecast with the new prediction
        new_forecast = np.append(forecast[0, 1:, 0], pred).reshape(1, 100, 1)
        forecast = new_forecast

    return predictions, lower_bounds, upper_bounds

# Fetch new data and preprocess it
final_df = pd.DataFrame(df['Close']) 

# Scaling data
scaler = MinMaxScaler(feature_range=(0, 1))
input_data = scaler.fit_transform(final_df)

# Prepare the last known data points
last_known_points = input_data[-100:]

# Forecasting the next 10 days with confidence intervals
predictions_10_days, lower_bounds, upper_bounds = multistep_forecast_with_confidence(model, scaler, last_known_points, 10)