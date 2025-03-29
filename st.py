# Streamlit Interface for Pairs Trading with Statistical Arbitrage and ML Predictive Modeling

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

sns.set(style="whitegrid")

# Functions
def download_data(ticker1, ticker2, start, end):
    asset1 = yf.download(ticker1, start=start, end=end)['Close']
    asset2 = yf.download(ticker2, start=start, end=end)['Close']
    df = pd.concat([asset1, asset2], axis=1).dropna()
    df.columns = [ticker1, ticker2]
    return df

def prepare_data(df):
    df['Spread'] = df.iloc[:,0] - df.iloc[:,1]
    df['Spread_lag'] = df['Spread'].shift(1)
    df.dropna(inplace=True)
    X = df[['Spread_lag']]
    y = df['Spread']
    return train_test_split(X, y, test_size=0.3, shuffle=False)

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    percentage_error = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    return predictions, mse, mae, percentage_error

def lstm_model(X_train, X_test, y_train, y_test):
    X_train_lstm = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_lstm, y_train, epochs=20, verbose=0)
    predictions = model.predict(X_test_lstm).flatten()
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    percentage_error = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    return predictions, mse, mae, percentage_error

# Streamlit Interface
st.title('Pairs Trading with Statistical Arbitrage and ML')

# User Inputs
ticker1 = st.text_input('Enter first stock ticker:', 'MSFT')
ticker2 = st.text_input('Enter second stock ticker:', 'AAPL')
start_date = st.date_input('Start Date', value=pd.to_datetime('2022-01-01'))
end_date = st.date_input('End Date', value=pd.to_datetime('2024-01-01'))

if st.button('Analyze and Predict'):
    df = download_data(ticker1, ticker2, start_date, end_date)
    X_train, X_test, y_train, y_test = prepare_data(df)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'Support Vector Regression': SVR()
    }

    results = {}
    for name, model in models.items():
        predictions, mse, mae, perc_error = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[name] = {'MSE': mse, 'MAE': mae, 'Percentage Error (%)': perc_error}
        st.subheader(f'{name} Results')
        st.write(pd.DataFrame({'Actual Spread': y_test, 'Predicted Spread': predictions}))

    predictions, mse, mae, perc_error = lstm_model(X_train, X_test, y_train, y_test)
    results['LSTM'] = {'MSE': mse, 'MAE': mae, 'Percentage Error (%)': perc_error}
    st.subheader('LSTM Results')
    st.write(pd.DataFrame({'Actual Spread': y_test, 'Predicted Spread': predictions}))

    # Comparative Results
    results_df = pd.DataFrame(results).T
    st.subheader('Comparative Analysis of ML Models')
    st.write(results_df)

    # Visualization
    st.bar_chart(results_df['Percentage Error (%)'])
