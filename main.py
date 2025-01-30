import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def fetch_and_train():
    # Fetch stock data from Yahoo Finance
    stock_data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')

    # Calculate 50-day and 200-day Simple Moving Averages (SMA)
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()

    # Drop rows with NaN values that were created due to rolling averages
    stock_data = stock_data.dropna()

    # Features (Close Price and SMAs)
    X = stock_data[['Close', 'SMA_50', 'SMA_200']]

    # Target (Next day's closing price)
    y = stock_data['Close'].shift(-1).dropna()  # Shift to predict the next day

    # Align X and y after shifting
    X = X.iloc[:-1]

    # Train/Test split xGradientBoostingRegressor
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False,train_size=0.8)

    # Model: Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

if __name__ == '__main__':
    mse = fetch_and_train()
    print(f'Mean Squared Error: {mse}')