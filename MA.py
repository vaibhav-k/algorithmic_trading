"""
This code loads historical price data from a CSV file,
calculates moving averages, and
determines trading signals based on whether the
short-term moving average is greater than the long-term moving average.
It then applies the trading strategy to calculate daily returns and cumulative returns.
Finally, it plots the cumulative returns of the trading strategy versus a buy and hold strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load historical price data
df = pd.read_csv('AAPL.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Calculate moving averages
df['MA_20'] = df['Close'].rolling(window=20).mean()
df['MA_50'] = df['Close'].rolling(window=50).mean()

# Determine trading signals
df['Signal'] = np.where(df['MA_20'] > df['MA_50'], 1, -1)

# Calculate daily returns
df['Returns'] = df['Close'].pct_change()

# Apply trading strategy
df['Strategy'] = df['Signal'].shift(1) * df['Returns']

# Calculate cumulative returns
df['Cumulative Returns'] = (1 + df['Strategy']).cumprod()

# Plot results
fig, ax = plt.subplots()
ax.plot(df.index, df['Cumulative Returns'], label='Strategy')
ax.plot(df.index, (1 + df['Returns']).cumprod(), label='Buy and Hold')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Returns')
ax.legend()
plt.show()
