"""
se the yfinance library to download historical stock data for Apple (AAPL) between January 1, 2022, and December 31, 2022.
We also use the newsapi library to download news articles for the same time period.
We, then, use the SentimentIntensityAnalyzer from the nltk library to analyze the sentiment of the news articles and
convert the sentiment scores to trading signals. When the sentiment score is above a certain threshold (0.2 in this case),
we generate a buy signal, and when it is below the threshold, we generate a sell signal.
We, then, calculate the daily returns and use the trading signals to calculate the strategy returns.
Finally, we calculate the cumulative returns of the strategy and plot them.
"""

from newsapi.newsapi_client import NewsApiClient
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime as dt

import nltk
nltk.download('vader_lexicon')

# Define the trading period
start_date = '2023-02-11'
end_date = '2023-03-11'

# Define the stock ticker symbol
ticker = 'AAPL'

# Download the historical data for the stock
data = yf.download(ticker, start=start_date, end=end_date)

# Download news articles for the stock
newsapi = NewsApiClient(api_key='7da2e9ec591748f2acae878a6eb09fe6')
news = newsapi.get_everything(
    q=ticker, from_param=start_date, to=end_date, language='en')

# Analyze the sentiment of the news articles
sia = SentimentIntensityAnalyzer()
sentiment_scores = []
for article in news['articles']:
    sentiment_scores.append(sia.polarity_scores(article['title'])['compound'])

# Convert the sentiment scores to trading signals
threshold = 0.2
data.dropna(inplace=True)
data['signal'] = pd.Series(
    np.where(np.array(sentiment_scores) > threshold, 1, 0))

# Calculate the daily returns
data['return'] = data['Close'].pct_change()

# Calculate the strategy returns
data['strategy_return'] = data['signal'].shift(1) * data['return']

# Calculate the cumulative returns
data['cumulative_return'] = (1 + data['strategy_return']).cumprod()

# Plot the cumulative returns
data['cumulative_return'].plot()
plt.show()
