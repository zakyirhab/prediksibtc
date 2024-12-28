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
import talib as ta  
import joblib
from datetime import datetime

# Function to create dataset
def create_dataset(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def predict_future_lstm(model, input_data, num_days, scaler, sequence_length):
    predictions = []
    current_input = input_data.copy()

    for _ in range(num_days):
        # Reshape input for LSTM
        current_input_reshaped = np.reshape(current_input, (1, sequence_length, current_input.shape[-1]))
        predicted = model.predict(current_input_reshaped)[0]

        # Append prediction
        predictions.append(predicted)

        # Update input with the new prediction
        new_row = np.zeros((1, current_input.shape[-1]))  # Placeholder for EMA, MA (if applicable)
        new_row[0, 0] = predicted  # Only the Close price is predicted
        current_input = np.vstack([current_input[1:], new_row])  # Shift window

    # Scale back to original values
    predictions = np.array(predictions).reshape(-1, 1)
    predictions_extended = np.zeros((predictions.shape[0], current_input.shape[-1]))
    predictions_extended[:, 0] = predictions[:, 0]
    predictions_original_scale = scaler.inverse_transform(predictions_extended)[:, 0]

    return predictions_original_scale

def predict_future_knn(model, input_data, num_days, scaler, sequence_length):
    predictions = []
    current_input = input_data.copy().reshape(-1, 3)  # Ensure correct shape for KNN

    for _ in range(num_days):
        # Reshape input for KNN
        current_input_reshaped = current_input.reshape(1, -1)
        predicted = model.predict(current_input_reshaped)[0]

        # Append prediction
        predictions.append(predicted)

        # Update input with the new prediction
        new_row = np.zeros((1, current_input.shape[1]))  # Placeholder for EMA, MA (if applicable)
        new_row[0, 0] = predicted  # Only the Close price is predicted
        current_input = np.vstack([current_input[1:], new_row])  # Shift window

    # Scale back to original values
    predictions = np.array(predictions).reshape(-1, 1)
    predictions_extended = np.zeros((predictions.shape[0], 3))  # Ensure correct shape for scaler
    predictions_extended[:, 0] = predictions[:, 0]
    predictions_original_scale = scaler.inverse_transform(predictions_extended)[:, 0]

    return predictions_original_scale

# Load models
knn_model = joblib.load('/content/best_knn_model.pkl')  # path to your saved KNN model
lstm_model = load_model('/content/model_bitcoin.h5')  # path to your saved LSTM model

# Streamlit user interface
st.title('Bitcoin Price Prediction')

# Sidebar for navigation
page = st.sidebar.radio("Select Page", ["Historical Data Prediction", "Future Prediction with LSTM", "Future Prediction with KNN"])

if page == "Historical Data Prediction":
    st.header('Historical Data Prediction')
    model_option = st.selectbox('Choose a model for prediction', ['LSTM', 'KNN'])
    ticker = st.text_input('Ticker Symbol', 'BTC-USD')
    start_date = st.date_input('Start Date', pd.to_datetime('2015-01-01'))
    end_date = st.date_input('End Date', pd.to_datetime('2024-01-01'))
    sequence_length = 60

    if st.button('Predict'):
        # Fetch data
        btc_data = yf.download(ticker, start=start_date, end=end_date)
        btc_data.columns = btc_data.columns.get_level_values(0)

        # Calculate indicators
        btc_data['EMA'] = ta.EMA(btc_data['Close'], timeperiod=100)  # Menggunakan talib
        btc_data['MA'] = ta.SMA(btc_data['Close'], timeperiod=200)  # Menggunakan talib
        btc_data = btc_data.dropna()
        data_to_scale = btc_data[['Close', 'EMA', 'MA']]

        # Scale data
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
            # Predict prices with KNN
            predicted_prices = knn_model.predict(X)
            predicted_prices_extended = np.zeros((predicted_prices.shape[0], 3))
            predicted_prices_extended[:, 0] = predicted_prices

        # Inverse transform to get actual prices
        predicted_prices_original_scale = scaler.inverse_transform(predicted_prices_extended)[:, 0]
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
        plt.figure(figsize=(12, 6))
        plt.plot(btc_data.index[sequence_length:], y_actual_original_scale, label='Harga Aktual', color='blue', linestyle='-')
        plt.plot(btc_data.index[sequence_length:], predicted_prices_original_scale, label='Harga Prediksi', color='red', linestyle='--')
        plt.title('Bitcoin Price Prediction on New Data')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

elif page == "Future Prediction with LSTM":
    st.header('Future Prediction with LSTM')
    ticker = st.text_input('Ticker Symbol', 'BTC-USD')
    start_date = datetime.now().date()  # Automatically set to current date
    st.write(f'Start Date: {start_date}')  # Display current date
    end_date = st.date_input('End Date', pd.to_datetime('2025-01-01'))  # User input for end date
    sequence_length = 60
    
    if st.button('Predict'):
        # Fetch historical data (fixed range)
        historical_start_date = '2015-01-01'
        historical_end_date = '2024-01-01'
        btc_data = yf.download(ticker, start=historical_start_date, end=historical_end_date)
        btc_data.columns = btc_data.columns.get_level_values(0)

        # Calculate indicators
        btc_data['EMA'] = ta.EMA(btc_data['Close'], timeperiod=100)  # Menggunakan talib
        btc_data['MA'] = ta.SMA(btc_data['Close'], timeperiod=200)  # Menggunakan talib
        btc_data = btc_data.dropna()
        data_to_scale = btc_data[['Close', 'EMA', 'MA']]

        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_to_scale)
        
        # Create dataset
        X, y = create_dataset(scaled_data, sequence_length)
        X = X.reshape(X.shape[0], sequence_length, 3)

        # Calculate number of days to predict based on end date
        num_days = (end_date - start_date).days

        # Predict future prices using LSTM
        future_predictions = predict_future_lstm(
            model=lstm_model,
            input_data=X[-1],  # Last sequence from the dataset
            num_days=num_days,
            scaler=scaler,
            sequence_length=sequence_length
        )

        # Display future predictions
        st.write(f'Future Predictions from {start_date} to {end_date}:')
        st.write(future_predictions)

        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(future_predictions)), future_predictions, label='Future Predictions', color='green')
        plt.title('Future Prediction with LSTM')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)

if page == "Future Prediction with KNN":
    st.header('Future Prediction with KNN')
    ticker = st.text_input('Ticker Symbol', 'BTC-USD')
    start_date = datetime.now().date()  # Automatically set to current date
    st.write(f'Start Date: {start_date}')  # Display current date
    end_date = st.date_input('End Date', pd.to_datetime('2025-01-01'))  # User input for end date
    sequence_length = 60
    
    if st.button('Predict'):
        # Fetch historical data (fixed range)
        historical_start_date = '2015-01-01'
        historical_end_date = '2024-01-01'
        btc_data = yf.download(ticker, start=historical_start_date, end=historical_end_date)
        btc_data.columns = btc_data.columns.get_level_values(0)

        # Calculate indicators
        btc_data['EMA'] = ta.EMA(btc_data['Close'], timeperiod=100)  # Menggunakan talib
        btc_data['MA'] = ta.SMA(btc_data['Close'], timeperiod=200)  # Menggunakan talib
        btc_data = btc_data.dropna()
        data_to_scale = btc_data[['Close', 'EMA', 'MA']]

        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_to_scale)
        
        # Create dataset
        X, y = create_dataset(scaled_data, sequence_length)
        X = np.reshape(X, (X.shape[0], -1))

        # Calculate number of days to predict based on end date
        num_days = (end_date - start_date).days

        # Ensure X[-1] has the correct shape for KNN
        input_data = X[-1].reshape(-1)

        # Predict future prices using KNN
        future_predictions = predict_future_knn(
            model=knn_model,
            input_data=input_data,  # Last sequence from the dataset
            num_days=num_days,
            scaler=scaler,
            sequence_length=sequence_length
        )

        # Display future predictions
        st.write(f'Future Predictions from {start_date} to {end_date}:')
        st.write(future_predictions)

        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(future_predictions)), future_predictions, label='Future Predictions', color='green')
        plt.title('Future Prediction with KNN')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)
