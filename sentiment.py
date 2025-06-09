import yfinance as yf#downloads the stock market data
import pandas as pd#handels the tabular data
import matplotlib.pyplot as plt#plot charts
import seaborn as sns
from textblob import TextBlob#analize the sentiment of text(positive or negative)


# Step 1: Download Apple stock price with auto_adjust = True
ticker = 'AAPL'
data = yf.download(ticker, start="2023-01-01", end="2024-12-31", auto_adjust=True)

# Ensure both index and columns are single-level
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Reset index to flatten Date
data = data.reset_index()

# Step 2: Sample financial headlines (mock)
headlines = [
    "Apple reports record revenue and strong iPhone sales.",
    "Apple faces supply chain issues amid China lockdown.",
    "New Apple Watch launch receives positive reviews.",
    "Regulators investigate Apple over App Store policies.",
    "Apple stock downgraded due to valuation concerns.",
    "Apple plans to expand services segment aggressively.",
    "Weak demand for iPads reported by Apple suppliers.",
    "Apple to announce new AI features in WWDC."
]
dates = pd.date_range("2023-01-05", periods=len(headlines), freq="15D")
news_df = pd.DataFrame({'Date': dates, 'Headline': headlines})

# Step 3: Sentiment scoring using TextBlob
news_df['Sentiment'] = news_df['Headline'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Ensure Date columns are datetime
data['Date'] = pd.to_datetime(data['Date'])
news_df['Date'] = pd.to_datetime(news_df['Date'])

# Sort before merge_asof
data_sorted = data.sort_values("Date")
news_sorted = news_df.sort_values("Date")

# Step 4: Merge sentiment data into stock data
merged_df = pd.merge_asof(news_sorted, data_sorted, on="Date")

# Step 5: Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=merged_df, x='Date', y='Close', label='AAPL Close Price')
sns.scatterplot(data=merged_df, x='Date', y='Close', size='Sentiment', hue='Sentiment',
                palette='coolwarm', legend=False)
plt.title('Apple Stock Price vs. News Sentiment')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.grid(True)
plt.tight_layout()
plt.show()
