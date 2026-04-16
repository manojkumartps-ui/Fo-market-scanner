import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# 1. Setup / Lexicon Download
@st.cache_resource
def setup_nlp():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = setup_nlp()

# 2. Logic: Full F&O List Building
@st.cache_data(ttl=86400) # Cache for 24 hours (List only changes daily)
def get_full_fno_list():
    try:
        # Fetch live instrument list from Zerodha Public API
        url = "https://api.kite.trade/instruments"
        df = pd.read_csv(url)
        # Filter for NFO segment and Futures to get a clean list of underlyings
        fno_df = df[df['segment'] == 'NFO-FUT']
        return sorted(fno_df['name'].unique().tolist())
    except Exception as e:
        st.error(f"Failed to fetch F&O list: {e}")
        return ["NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK", "INFY"]

# 3. Logic: Crowd Sentiment Scoring (Finviz/News Scraper)
def get_sentiment(ticker):
    try:
        url = f'https://finviz.com{ticker}'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        news_table = soup.find(id='news-table')
        if not news_table: return 0.0
        
        headlines = [row.a.text for row in news_table.findAll('tr')[:10]]
        scores = [sia.polarity_scores(h)['compound'] for h in headlines]
        return round(sum(scores) / len(scores), 4) if scores else 0.0
    except:
        return 0.0

# 4. Logic: Heikin Ashi Trend Scoring (Mock Calculation)
def get_ha_trend_score(ticker):
    """
    In a real app, you'd fetch OHLC data here. 
    Logic: Green HA = +1, Red HA = -1, Strong (no wick) = +/-2.
    """
    # Placeholder for trend logic (Integrate with yfinance/kite for live prices)
    return "Bullish 🟢" # Default for display

# --- STREAMLIT UI ---
st.set_page_config(page_title="NSE F&O Scanner", layout="wide")
st.title("🚀 NSE F&O Market Scanner")

# Sidebar settings
scan_limit = st.sidebar.slider("Number of stocks to scan", 5, 50, 20)

if st.button("🔄 Run Full Market Scan"):
    fno_list = get_full_fno_list()
    scan_list = fno_list[:scan_limit]
    
    results = []
    progress_bar = st.progress(0)
    
    for i, symbol in enumerate(scan_list):
        sentiment_score = get_sentiment(symbol)
        trend = get_ha_trend_score(symbol)
        
        results.append({
            "Symbol": symbol,
            "Sentiment Score": sentiment_score,
            "Crowd Sentiment": "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral",
            "HA Trend": trend
        })
        progress_bar.progress((i + 1) / len(scan_list))
    
    # Final Presentation
    df_results = pd.DataFrame(results)
    
    # Use columns to show key stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Scanned", len(results))
    c2.metric("Bullish Sentiment", len(df_results[df_results['Sentiment Score'] > 0.05]))
    c3.metric("Bearish Sentiment", len(df_results[df_results['Sentiment Score'] < -0.05]))

    st.dataframe(df_results.style.background_gradient(subset=['Sentiment Score'], cmap='RdYlGn'), use_container_width=True)
