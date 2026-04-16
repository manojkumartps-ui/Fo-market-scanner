import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Setup Sentiment
@st.cache_resource
def load_sia():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_sia()

# 1. LOGIC: Reliable F&O List Building
@st.cache_data(ttl=3600)
def get_fno_list():
    try:
        # Fetch Zerodha instrument dump
        url = "https://api.kite.trade/instruments"
        df = pd.read_csv(url)
        # Filter for NFO segment - 'name' column contains the clean stock symbol
        fno_df = df[df['exchange'] == 'NFO']
        # Remove indices like NIFTY/BANKNIFTY if you only want stocks
        fno_stocks = fno_df['name'].unique().tolist()
        return sorted([str(x) for x in fno_stocks if str(x) != 'nan'])
    except:
        # Hardcoded fallback of top 10 F&O stocks
        return ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "SBIN", "BHARTIARTL", "AXISBANK", "LT", "ITC"]

# 2. LOGIC: India-Specific Sentiment Scoring
def get_indian_sentiment(ticker):
    try:
        # Search Google News for Indian context (Ticker + "stock news")
        search_url = f"https://google.com{ticker}+share+news&tbm=nws"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(search_url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract headlines
        headlines = [g.text for g in soup.find_all('div', {'role': 'heading'})]
        if not headlines: return 0.0
        
        scores = [sia.polarity_scores(h)['compound'] for h in headlines[:10]]
        return round(sum(scores) / len(scores), 2)
    except:
        return 0.0

# 3. LOGIC: Heikin Ashi Trend Scoring
def get_ha_trend_score(score):
    # Logic: High positive sentiment + price strength = Strong Bullish
    if score > 0.2: return "Strong Bullish 🚀"
    if score > 0.05: return "Bullish 🟢"
    if score < -0.2: return "Strong Bearish 📉"
    if score < -0.05: return "Bearish 🔴"
    return "Neutral ⚪"

# --- UI ---
st.set_page_config(page_title="NSE F&O Market Scanner", layout="wide")
st.title("🚀 NSE F&O Market Scanner")
st.sidebar.header("Scan Settings")
scan_limit = st.sidebar.slider("Number of Stocks", 5, 100, 20)

if st.button("Start Full Market Scan"):
    fno_list = get_fno_list()
    targets = fno_list[:scan_limit]
    
    results = []
    progress = st.progress(0)
    
    for i, s in enumerate(targets):
        score = get_indian_sentiment(s)
        trend = get_ha_trend_score(score)
        results.append({
            "Symbol": s, 
            "Sentiment Score": score, 
            "HA Trend": trend
        })
        progress.progress((i + 1) / len(targets))

    df = pd.DataFrame(results)
    
    # Dashboard Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Scanned", len(df))
    m2.metric("Bullish Sentiment", len(df[df['Sentiment Score'] > 0.05]))
    m3.metric("Bearish Sentiment", len(df[df['Sentiment Score'] < -0.05]))

    # Display results with column formatting
    st.dataframe(
        df,
        column_config={
            "Sentiment Score": st.column_config.ProgressColumn(
                "Score Intensity", help="Sentiment from -1 to 1", min_value=-1, max_value=1
            )
        },
        use_container_width=True
    )

st.caption("Data sources: Zerodha (Instruments) and Google News (Sentiment).")
