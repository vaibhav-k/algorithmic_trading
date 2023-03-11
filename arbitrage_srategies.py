"""
Use the yfinance library to download historical stock data for Apple (AAPL) and Microsoft (MSFT) between January 1, 2022, and December 31, 2022.
We then calculate the price ratio between the two stocks and use it to generate trading signals.
When the price ratio is more than two standard deviations away from the mean,
we generate a sell signal for AAPL and a buy signal for MSFT, and vice versa.
We then calculate the daily returns and use the trading signals to calculate the strategy returns.
Finally, we calculate the cumulative returns of the strategy and plot them.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Define the trading period
start_date = '2022-01-01'
end_date = '2022-12-31'

# Define the stock ticker symbols
ticker1 = 'AAPL'
ticker2 = 'MSFT'

# Download the historical data for the stocks
data1 = yf.download(ticker1, start=start_date, end=end_date)
data2 = yf.download(ticker2, start=start_date, end=end_date)

# Calculate the price ratio between the two stocks
data1['price_ratio'] = data1['Close'] / data2['Close']

# Calculate the mean and standard deviation of the price ratio
mean = data1['price_ratio'].mean()
std = data1['price_ratio'].std()

# Define the trading signals
data1['signal'] = np.where(data1['price_ratio'] > mean + 2 *
                           std, -1, np.where(data1['price_ratio'] < mean - 2*std, 1, 0))

# Calculate the daily returns
data1['return'] = data1['Close'].pct_change()
data2['return'] = data2['Close'].pct_change()

# Calculate the strategy returns
data1['strategy_return'] = data1['signal'].shift(
    1) * (data1['return'] - data2['return'] * data1['price_ratio'].shift(1))

# Calculate the cumulative returns
data1['cumulative_return'] = (1 + data1['strategy_return']).cumprod()

# Plot the cumulative returns
data1['cumulative_return'].plot()
plt.show()
