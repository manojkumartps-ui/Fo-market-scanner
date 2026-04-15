import pandas as pd
from flask import Flask, render_template_string
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Initialize Sentiment Analyzer
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

app = Flask(__name__)

# Logic 1: Build the F&O List
def get_fno_list():
    try:
        url = "https://kite.trade"
        df = pd.read_csv(url)
        # Filter for NFO (NSE Futures & Options) segment
        fno_df = df[df['exchange'] == 'NFO']
        return sorted(fno_df['name'].unique().tolist())
    except Exception as e:
        return []

# Logic 2: Crowd-Sourced Sentiment & Scoring
def get_sentiment_score(ticker):
    """
    Scrapes recent headlines for a ticker and calculates a composite score.
    Logic: Compound score > 0.05 is Bullish, < -0.05 is Bearish.
    """
    try:
        # Example using Finviz for sentiment (Common community source)
        url = f'https://finviz.com{ticker}'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Logic: Find the news table
        news_table = soup.find(id='news-table')
        if not news_table: return 0.0 # Neutral if no news
        
        headlines = [row.a.text for row in news_table.findAll('tr')]
        scores = [sia.polarity_scores(h)['compound'] for h in headlines[:10]]
        
        # Return average sentiment score
        return round(sum(scores) / len(scores), 2) if scores else 0.0
    except:
        return 0.0

# Logic 3: Heikin Ashi Calculation
def calculate_heikin_ashi(df):
    ha_df = df.copy()
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    ha_open = [(df['Open'].iloc[0] + df['Close'].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2)
    
    ha_df['HA_Open'] = ha_open
    ha_df['HA_High'] = ha_df[['HA_Open', 'HA_Close', 'High']].max(axis=1)
    ha_df['HA_Low'] = ha_df[['HA_Open', 'HA_Close', 'Low']].min(axis=1)
    return ha_df

@app.route('/')
def dashboard():
    stocks = get_fno_list()[:15] # Limiting to 15 for speed
    results = []
    
    for symbol in stocks:
        score = get_sentiment_score(symbol)
        sentiment_label = "Bullish" if score > 0.05 else "Bearish" if score < -0.05 else "Neutral"
        results.append({"symbol": symbol, "score": score, "sentiment": sentiment_label})

    html = """
    <body style="font-family: sans-serif; padding: 20px;">
        <h1>NSE F&O Sentiment Dashboard</h1>
        <table border="1" cellpadding="10" style="border-collapse: collapse;">
            <tr><th>Symbol</th><th>Sentiment Score (-1 to 1)</th><th>Crowd Bias</th></tr>
            {% for item in data %}
            <tr>
                <td>{{ item.symbol }}</td>
                <td style="color: {{ 'green' if item.score > 0 else 'red' }}">{{ item.score }}</td>
                <td><b>{{ item.sentiment }}</b></td>
            </tr>
            {% endfor %}
        </table>
    </body>
    """
    return render_template_string(html, data=results)

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
