"""
Use the yfinance library to download historical stock data for Apple (AAPL) between January 1, 2022, and December 31,
2022. We then calculate the 20-day moving average and standard deviation and use them to calculate the upper and lower bands for mean-reversion.
When the price of the stock crosses the upper band, we generate a sell signal, and when it crosses the lower band, we generate a buy signal.
We then calculate the daily returns and use the trading signals to calculate the strategy returns.
Finally, we calculate the cumulative returns of the strategy and plot them.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

pd.options.mode.chained_assignment = None

# Define the trading period
start_date = '2022-01-01'
end_date = '2022-12-31'

# Define the stock ticker symbol
ticker = 'AAPL'

# Download the historical data for the stock
data = yf.download(ticker, start=start_date, end=end_date)

# Calculate the 20-day moving average and standard deviation
data['MA'] = data['Close'].rolling(window=20).mean()
data['STD'] = data['Close'].rolling(window=20).std()

# Calculate the upper and lower bands
data['Upper_Band'] = data['MA'] + (data['STD'] * 2)
data['Lower_Band'] = data['MA'] - (data['STD'] * 2)

# Define the trading signals
data['signal'] = 0
data['signal'][20:] = np.where(
    data['Close'][20:] > data['Upper_Band'][20:], -1, 0)
data['signal'][20:] = np.where(
    data['Close'][20:] < data['Lower_Band'][20:], 1, 0)

# Calculate the daily returns
data['return'] = data['Close'].pct_change()

# Calculate the strategy returns
data['strategy_return'] = data['signal'].shift(1) * data['return']

# Calculate the cumulative returns
data['cumulative_return'] = (1 + data['strategy_return']).cumprod()

# Plot the cumulative returns
data['cumulative_return'].plot()
plt.show()
