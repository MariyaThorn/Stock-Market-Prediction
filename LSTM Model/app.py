
import yfinance as yahooFinance
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Setting up the Streamlit app
st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
GetAppleInformation = yahooFinance.Ticker(user_input)
start = '2020-01-01'
end = '2024-03-31'
historical_data = GetAppleInformation.history(start=start, end=end)

# Describing Data
st.subheader('Data from 2020 to 2024')
st.write(historical_data.describe())

# Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(historical_data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = historical_data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, label='MA 100')
plt.plot(historical_data.Close, label='Closing Price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma200 = historical_data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, label='MA 100')
plt.plot(ma200, label='MA 200')
plt.plot(historical_data.Close, label='Closing Price')
plt.legend()
st.pyplot(fig)

# Data Preprocessing for Model
data_training = pd.DataFrame(historical_data['Close'][0:int(len(historical_data)*0.70)])
data_testing = pd.DataFrame(historical_data['Close'][int(len(historical_data)*0.70):])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Model Building
x_train = []
y_train = []

for i in range(100, len(data_training_array)):
    x_train.append(data_training_array[i-100:i, 0])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80))
model.add(Dropout(0.4))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Preparing Test Data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test = []
for i in range(100, len(input_data)):
    x_test.append(input_data[i-100:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_predicted = model.predict(x_test)
y_predicted = scaler.inverse_transform(y_predicted)

# Visualizing the results
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(data_testing.values, color='blue', label='Original Price')
plt.plot(y_predicted, color='red', label='Predicted Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
