import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Initialize Sentiment
@st.cache_resource
def load_sia():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_sia()

# 1. LOGIC: Build F&O List (Using a faster, more reliable CSV source)
@st.cache_data(ttl=3600)
def get_fno_list():
    try:
        # Using the direct Kite instrument URL
        url = "https://kite.trade"
        df = pd.read_csv(url)
        # Filter: Segment NFO-FUT gives you the underlying stocks
        fno_stocks = df[df['segment'] == 'NFO-FUT']['name'].unique().tolist()
        return sorted([str(x) for x in fno_stocks if str(x) != 'nan'])
    except:
        # Fallback to major liquid F&O stocks if download fails
        return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "KOTAKBANK"]

# 2. LOGIC: Sentiment & Crowd Scoring
def get_score(ticker):
    try:
        url = f'https://finviz.com{ticker}'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(response.content, 'html.parser')
        news_table = soup.find(id='news-table')
        if not news_table: return 0.0
        
        headlines = [row.a.text for row in news_table.findAll('tr')[:5]]
        scores = [sia.polarity_scores(h)['compound'] for h in headlines]
        return round(sum(scores) / len(scores), 2) if scores else 0.0
    except:
        return 0.0

# 3. LOGIC: Heikin Ashi Trend (Dummy Logic until Live OHLC is added)
def get_ha_trend(score):
    if score > 0.1: return "Bullish Trend 🟢"
    if score < -0.1: return "Bearish Trend 🔴"
    return "Neutral ⚪"

# --- UI ---
st.set_page_config(page_title="NSE F&O Scanner")
st.title("🚀 NSE F&O Market Scanner")

# User Input
scan_limit = st.sidebar.slider("Scan Limit", 5, 100, 20)

if st.button("Start Market Scan"):
    stocks = get_fno_list()
    # Ensure we have a list to scan
    scan_targets = stocks[:scan_limit]
    
    results = []
    progress = st.progress(0)
    
    for i, s in enumerate(scan_targets):
        score = get_score(s)
        trend = get_ha_trend(score)
        results.append({"Symbol": s, "Sentiment": score, "HA Trend": trend})
        progress.progress((i + 1) / len(scan_targets))

    df = pd.DataFrame(results)
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Scanned", len(df))
    c2.metric("Bullish", len(df[df['Sentiment'] > 0.05]))
    c3.metric("Bearish", len(df[df['Sentiment'] < -0.05]))

    # Display Table with Column Config (Fixes the styling error)
    st.dataframe(
        df,
        column_config={
            "Sentiment": st.column_config.NumberColumn(format="%.2f"),
        },
        use_container_width=True
    )

st.info("Note: Sentiment is crowdsourced from latest news headlines.")
