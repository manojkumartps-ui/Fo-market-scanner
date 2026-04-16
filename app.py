import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# 1. Setup / Requirements
@st.cache_resource
def load_nltk():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_nltk()

# 2. Logic: Build F&O List
@st.cache_data(ttl=3600) # Caches the list for 1 hour
def get_fno_list():
    try:
        url = "https://kite.trade"
        df = pd.read_csv(url)
        fno_df = df[df['exchange'] == 'NFO']
        return sorted(fno_df['name'].unique().tolist())
    except:
        return ["NIFTY", "BANKNIFTY"] # Fallback

# 3. Logic: Sentiment Scoring
def get_sentiment_score(ticker):
    try:
        url = f'https://finviz.com{ticker}'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        news_table = soup.find(id='news-table')
        if not news_table: return 0.0
        
        headlines = [row.a.text for row in news_table.findAll('tr')]
        scores = [sia.polarity_scores(h)['compound'] for h in headlines[:10]]
        return round(sum(scores) / len(scores), 2) if scores else 0.0
    except:
        return 0.0

# 4. Logic: Heikin Ashi (Transformation only)
def calculate_heikin_ashi(df):
    # This logic is ready for when you pass a Price DataFrame
    ha_df = df.copy()
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    # ... (Rest of HA logic from previous turn)
    return ha_df

# --- STREAMLIT UI ---
st.title("🚀 NSE F&O Market Scanner")
st.subheader("Crowd Sentiment & Trend Scoring")

with st.status("Building F&O List and fetching sentiment..."):
    stocks = get_fno_list()[:20] # Scan top 20 for speed
    data = []
    for s in stocks:
        score = get_sentiment_score(s)
        data.append({
            "Symbol": s,
            "Score": score,
            "Sentiment": "Bullish 🟢" if score > 0.05 else "Bearish 🔴" if score < -0.05 else "Neutral ⚪"
        })

# Display as a clean table
st.table(pd.DataFrame(data))

st.success(f"Scanned {len(data)} stocks successfully.")
