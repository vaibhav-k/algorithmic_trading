"""
Use the yfinance library to download historical stock data for Apple (AAPL) between January 1, 2022, and March 10, 2022.
We, then, calculate the 50-day and 200-day moving averages and use them to generate trading signals.
When the 50-day moving average crosses above the 200-day moving average, we generate a buy signal,
and when it crosses below, we generate a sell signal.
We then calculate the daily returns and use the trading signals to calculate the strategy returns.
Finally, we calculate the cumulative returns of the strategy and plot them.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Define the trading period
start_date = '2022-01-01'
end_date = '2022-03-10'

# Define the stock ticker symbol
ticker = 'AAPL'

# Download the historical data for the stock
data = yf.download(ticker, start=start_date, end=end_date)

# Calculate the 50-day and 200-day moving averages
data['ma50'] = data['Close'].rolling(window=50).mean()
data['ma200'] = data['Close'].rolling(window=200).mean()

# Define the trading signals
data['signal'] = np.where(data['ma50'] > data['ma200'], 1, -1)

# Calculate the daily returns
data['return'] = data['Close'].pct_change()

# Calculate the strategy returns
data['strategy_return'] = data['signal'].shift(1) * data['return']

# Calculate the cumulative returns
data['cumulative_return'] = (1 + data['strategy_return']).cumprod()

# Plot the cumulative returns
data['cumulative_return'].plot()
plt.show()
