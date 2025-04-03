# Import required libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
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

sns.set(style="darkgrid")

# Function to download stock data
def download_data(ticker1, ticker2, start, end):
    asset1 = yf.download(ticker1, start=start, end=end)['Close']
    asset2 = yf.download(ticker2, start=start, end=end)['Close']
    df = pd.concat([asset1, asset2], axis=1).dropna()
    df.columns = [ticker1, ticker2]
    return df

# Perform the cointegration test
def cointegration_test(df, ticker1, ticker2):
    score, p_value, _ = ts.coint(df[ticker1], df[ticker2])
    print(f'Cointegration Test p-value: {p_value:.4f}')
    return p_value < 0.5  # If p-value < 0.05, the pair is cointegrated

# Prepare data and compute Z-score
def prepare_data(df):
    df['Spread'] = df.iloc[:,0] - df.iloc[:,1]
    df['Spread_lag'] = df['Spread'].shift(1)

    # Compute Z-Score for Spread
    df['Mean_Spread'] = df['Spread'].rolling(window=20).mean()
    df['Std_Spread'] = df['Spread'].rolling(window=20).std()
    df['Z-Score'] = (df['Spread'] - df['Mean_Spread']) / df['Std_Spread']
    
    df.dropna(inplace=True)
    X = df[['Spread_lag']]
    y = df['Spread']
    
    return train_test_split(X, y, test_size=0.3, shuffle=False), df

# Generate trading signals using Z-score
def generate_trading_signals(df):
    df['Long_Signal'] = df['Z-Score'] < -1  # Buy Signal (Spread is too low)
    df['Short_Signal'] = df['Z-Score'] > 1   # Sell Signal (Spread is too high)
    return df

# Evaluate ML models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    percentage_error = np.mean(np.abs((y_test - predictions) / y_test)) * 100

    results_df = pd.DataFrame({'Actual Spread': y_test, 'Predicted Spread': predictions}, index=y_test.index)
    print(f"\n{model_name} Predictions vs Actual:")
    print(results_df.head(10))

    plt.figure(figsize=(12,6))
    plt.plot(y_test.index, y_test, label='Actual Spread', color='blue')
    plt.plot(y_test.index, predictions, label=f'Predicted Spread ({model_name})', color='orange')
    plt.title(f'{model_name} Predicted vs Actual Spread')
    plt.xlabel('Date')
    plt.ylabel('Spread')
    plt.legend()
    plt.show()

    print(f'{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, Percentage Error: {percentage_error:.2f}%')
    return mse, mae, percentage_error

# LSTM model
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

    results_df = pd.DataFrame({'Actual Spread': y_test, 'Predicted Spread': predictions}, index=y_test.index)
    print("\nLSTM Predictions vs Actual:")
    print(results_df.head(10))

    plt.figure(figsize=(12,6))
    plt.plot(y_test.index, y_test, label='Actual Spread', color='blue')
    plt.plot(y_test.index, predictions, label='Predicted Spread (LSTM)', color='green')
    plt.title('LSTM Predicted vs Actual Spread')
    plt.xlabel('Date')
    plt.ylabel('Spread')
    plt.legend()
    plt.show()

    print(f'LSTM - MSE: {mse:.4f}, MAE: {mae:.4f}, Percentage Error: {percentage_error:.2f}%')
    return mse, mae, percentage_error

# Main execution
if __name__ == '__main__':
    ticker1 = 'AAL'  
    ticker2 = 'DAL'  
  # Chevron
    df = download_data(ticker1, ticker2, '2022-01-01', '2024-01-01')

    # Check for cointegration before proceeding
    if not cointegration_test(df, ticker1, ticker2):
        print(f'{ticker1} and {ticker2} are NOT cointegrated. Choose another pair!')
    else:
        print(f'{ticker1} and {ticker2} are cointegrated. Proceeding with analysis...')

        (X_train, X_test, y_train, y_test), df = prepare_data(df)
        df = generate_trading_signals(df)

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
            'Support Vector Regression': SVR()
        }

        results = {}
        for name, model in models.items():
            mse, mae, perc_error = evaluate_model(model, X_train, X_test, y_train, y_test, name)
            results[name] = {'MSE': mse, 'MAE': mae, 'Percentage Error (%)': perc_error}

        mse, mae, perc_error = lstm_model(X_train, X_test, y_train, y_test)
        results['LSTM'] = {'MSE': mse, 'MAE': mae, 'Percentage Error (%)': perc_error}

        results_df = pd.DataFrame(results).T
        print("\nComparative Analysis of ML Models:")
        print(results_df)

        results_df[['Percentage Error (%)']].plot(kind='bar', figsize=(10,6), title='Percentage Error Comparison among Models')
        plt.ylabel('Percentage Error (%)')
        plt.xlabel('ML Models')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
