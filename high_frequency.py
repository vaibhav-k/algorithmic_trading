"""
Use the yfinance library to download historical stock data for Apple (AAPL) between January 1, 2022, and December 31,
2022. We then define the trading parameters, including the buy and sell thresholds, and initialize the trading position and cash balance.
We then run a loop over the historical data and check if the price difference between the current and previous time periods exceeds the buy or sell threshold.
If it does, we place a trade and update the position and cash balance.
If not, we update the cash balance based on the current position and stock price.
Finally, we calculate the final cash balance and position value and print the results.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import time

# Define the trading period
start_date = '2022-01-01'
end_date = '2022-12-31'

# Define the stock ticker symbol
ticker = 'AAPL'

# Download the historical data for the stock
data = yf.download(ticker, start=start_date, end=end_date)

# Define the trading parameters
buy_threshold = 0.01
sell_threshold = -0.01

# Initialize the trading position and cash balance
position = 0
cash = 100000

# Define the trading loop
for i in range(1, len(data)):
    # Calculate the price difference between the current and previous time periods
    price_diff = data['Close'][i] - data['Close'][i-1]

    # Check if the price difference exceeds the buy threshold and there is enough cash to buy
    if price_diff > buy_threshold and cash > data['Close'][i]:
        # Buy the stock and update the position and cash balance
        position += 1
        cash -= data['Close'][i]
        print(f'Bought {position} shares at {data["Close"][i]}')
        time.sleep(0.1)

    # Check if the price difference exceeds the sell threshold and there is a position to sell
    elif price_diff < sell_threshold and position > 0:
        # Sell the stock and update the position and cash balance
        position -= 1
        cash += data['Close'][i]
        print(f'Sold {position} shares at {data["Close"][i]}')
        time.sleep(0.1)

    # Update the cash balance based on the current position and stock price
    else:
        cash += position * data['Close'][i]

# Calculate the final cash balance and position value
final_cash = cash + position * data['Close'][-1]
position_value = position * data['Close'][-1]

# Print the final results
print(f'Final cash balance: {final_cash}')
print(f'Position value: {position_value}')
print(f'Total portfolio value: {final_cash + position_value}')
