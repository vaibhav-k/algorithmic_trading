"""
Use the yfinance library to download historical stock data for Apple (AAPL) between January 1, 2022, and December 31, 2022.
We, then, calculate the daily returns and use the 50-day and 200-day moving averages as features and
the sign of the daily return as the target variable.
We split the data into training and testing sets and train a Random Forest classifier on the training data.
We, then, make predictions on the testing set and calculate the accuracy of the model.
Finally, we use the trained classifier to generate trading signals and calculate the strategy returns and cumulative returns.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the trading period
start_date = '2022-01-01'
end_date = '2022-12-31'

# Define the stock ticker symbol
ticker = 'AAPL'

# Download the historical data for the stock
data = yf.download(ticker, start=start_date, end=end_date)

# Calculate the daily returns
data['return'] = data['Close'].pct_change()

# Define the features and target variable
data['ma50'] = data['Close'].rolling(window=50).mean()
data['ma200'] = data['Close'].rolling(window=200).mean()
data.dropna(inplace=True)

X = data[['ma50', 'ma200']]
y = np.where(data['return'] > 0, 1, -1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Generate trading signals based on the predictions
data['signal'] = np.where(clf.predict(X) > 0, 1, -1)

# Calculate the strategy returns
data['strategy_return'] = data['signal'].shift(1) * data['return']

# Calculate the cumulative returns
data['cumulative_return'] = (1 + data['strategy_return']).cumprod()

# Plot the cumulative returns
data['cumulative_return'].plot()
plt.show()
