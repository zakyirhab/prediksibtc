import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import pandas_ta as ta
import joblib

# Function to create dataset
def create_dataset(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Load models
knn_model = joblib.load('/content/best_knn_model.pkl')
lstm_model = load_model('/content/model_bitcoin.h5')

# Streamlit user interface
st.title('Bitcoin Price Prediction')
model_option = st.selectbox('Choose a model for prediction', ['LSTM', 'KNN'])
ticker = st.text_input('Ticker Symbol', 'BTC-USD')
start_date = st.date_input('Start Date', pd.to_datetime('2015-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('2024-01-01'))
sequence_length = 60

if st.button('Predict'):
    btc_data = yf.download(ticker, start=start_date, end=end_date)
    btc_data.columns = btc_data.columns.get_level_values(0)

    # Calculate indicators
    btc_data['EMA'] = ta.EMA(btc_data['Close'], timeperiod=100)
    btc_data['MA'] = ta.SMA(btc_data['Close'], timeperiod=200)
    btc_data = btc_data.dropna()
    data_to_scale = btc_data[['Close', 'EMA', 'MA']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_to_scale)

    # Create dataset
    X, y = create_dataset(scaled_data, sequence_length)
    X = X.reshape(X.shape[0], -1)

    if model_option == 'LSTM':
        X = np.reshape(X, (X.shape[0], sequence_length, 3))
        predicted_prices = lstm_model.predict(X)
        predicted_prices_extended = np.zeros((predicted_prices.shape[0], 3))
        predicted_prices_extended[:, 0] = predicted_prices[:, 0]
    elif model_option == 'KNN':
        predicted_prices = knn_model.predict(X)
        predicted_prices_extended = np.zeros((predicted_prices.shape[0], 3))
        predicted_prices_extended[:, 0] = predicted_prices

    # Inverse transform to get actual prices
    predicted_prices_original_scale = scaler.inverse_transform(predicted_prices_extended)[:, 0]

    # Reverse scale y
    y_extended = np.zeros((y.shape[0], 3))
    y_extended[:, 0] = y
    y_actual_original_scale = scaler.inverse_transform(y_extended)[:, 0]

    # Calculate metrics
    mse = mean_squared_error(y_actual_original_scale, predicted_prices_original_scale)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual_original_scale, predicted_prices_original_scale)
    mape = mean_absolute_percentage_error(y_actual_original_scale, predicted_prices_original_scale)
    accuracy = 100 - (mape * 100)

    st.write(f'Akurasi: {accuracy:.2f}%')
    st.write(f'MAPE: {mape * 100:.2f}%')
    st.write(f'Mean Squared Error (MSE): {mse}')
    st.write(f'Root Mean Squared Error (RMSE): {rmse}')
    st.write(f'Mean Absolute Error (MAE): {mae}')

    # Plot predictions
    plt.figure(figsize=(12,6))
    plt.plot(y_actual_original_scale, color='blue', label='Actual Bitcoin Price')
    plt.plot(predicted_prices_original_scale, color='red', label='Predicted Bitcoin Price')
    plt.title(f'Bitcoin Price Prediction using {model_option}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)
