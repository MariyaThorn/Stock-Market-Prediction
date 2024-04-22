import yfinance as yahooFinance
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Setting up the Streamlit app
st.title('Stock Trend Prediction')

# User input for stock ticker
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
stock_data = yahooFinance.Ticker(user_input)

# Date range for fetching data
start = '2019-01-01'
end = '2024-03-31'
historical_data = stock_data.history(start=start, end=end)

# Describing Data
st.subheader('Data from 2020 to 2024')
st.write(historical_data.describe())

# Display the basic Closing Price vs Time chart
st.subheader('Closing Price vs Time')
fig, ax = plt.subplots(figsize=(12, 6))
historical_data['Close'].plot(ax=ax, label='Close Price')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.title.set_text('Daily Closing Price')
st.pyplot(fig)

# Adding moving averages to the basic plot
st.subheader('Closing Price vs Time with 100MA & 200MA')
historical_data['100MA'] = historical_data['Close'].rolling(window=100).mean()
historical_data['200MA'] = historical_data['Close'].rolling(window=200).mean()
fig_ma, ax_ma = plt.subplots(figsize=(12, 6))
ax_ma.plot(historical_data['Close'], label='Close Price')
ax_ma.plot(historical_data['100MA'], label='100-Day MA', color='r')
ax_ma.plot(historical_data['200MA'], label='200-Day MA', color='g')
ax_ma.legend()
st.pyplot(fig_ma)

# Allow user to choose the model type
model_type = st.selectbox('Choose the model type', ['LSTM', 'Random Forest'])

if model_type == 'LSTM':
    # Data preparation for LSTM
    data_training = pd.DataFrame(historical_data['Close'][0:int(len(historical_data)*0.70)])
    data_testing = pd.DataFrame(historical_data['Close'][int(len(historical_data)*0.70):])
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    # LSTM Model setup
    x_train = []
    y_train = []

    for i in range(100, len(data_training_array)):
        x_train.append(data_training_array[i-100:i, 0])
        y_train.append(data_training_array[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model_lstm = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=60, return_sequences=True),
        Dropout(0.3),
        LSTM(units=80),
        Dropout(0.4),
        Dense(units=1)
    ])

    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    model_lstm.fit(x_train, y_train, epochs=25, batch_size=32)

    # Preparing test data for LSTM
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.transform(final_df)

    x_test = []
    for i in range(100, len(input_data)):
        x_test.append(input_data[i-100:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_predicted = model_lstm.predict(x_test)
    y_predicted = scaler.inverse_transform(y_predicted)

    # Display LSTM predictions
    st.subheader('LSTM Predictions vs Original')
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(data_testing.values, color='blue', label='Original Price')
    plt.plot(y_predicted, color='red', label='Predicted Price')
    plt.title('Stock Price Prediction by LSTM')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

elif model_type == 'Random Forest':
    # Data preparation for Random Forest
    historical_data['Tomorrow'] = historical_data['Close'].shift(-1)
    historical_data['Target'] = (historical_data['Tomorrow'] > historical_data['Close']).astype(int)
    historical_data = historical_data.dropna()

    # Split data into training and testing
    train = historical_data.iloc[:-100]
    test = historical_data.iloc[-100:]

    predictors = ['Close', 'Volume', 'Open', 'High', 'Low']
    model_rf = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    model_rf.fit(train[predictors], train['Target'])

    # Prediction using Random Forest
    test['Predictions'] = model_rf.predict(test[predictors])
    precision = precision_score(test['Target'], test['Predictions'])

    # Display RF model performance
    st.subheader('Random Forest Model Performance')
    st.write(f'Precision Score: {precision}')
    fig_rf, ax = plt.subplots()
    ax.plot(test['Close'], label='Close Price')
    ax.scatter(test.index, test['Predictions'] * max(test['Close']), color='red', label='Predictions (scaled)')
    ax.legend()
    st.pyplot(fig_rf)
