import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import time

# --- INITIALIZATION ---
@st.cache_resource
def setup_nlp():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = setup_nlp()

# --- LOGIC 1: BUILD F&O LIST ---
@st.cache_data(ttl=3600)
def get_fno_list():
    """Fetches the live list of NSE F&O stocks from public feeds."""
    try:
        url = "https://kite.trade"
        df = pd.read_csv(url)
        # NFO segment filter isolates derivatives
        fno_df = df[df['exchange'] == 'NFO']
        symbols = sorted(fno_df['name'].unique().tolist())
        return [str(s) for s in symbols if str(s) != 'nan']
    except Exception as e:
        st.error(f"List building failed: {e}")
        return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN", "ICICIBANK"]

# --- LOGIC 2: HEIKIN ASHI TREND ---
def calculate_ha_trend(symbol):
    """Fetches OHLC and returns a Heikin Ashi trend score."""
    try:
        ticker = f"{symbol}.NS"
        data = yf.download(ticker, period="5d", interval="1d", progress=False)
        if data.empty: return "NO_DATA", 0
        
        # Latest bar
        last = data.iloc[-1]
        prev = data.iloc[-2]
        
        # HA Close = (O+H+L+C)/4
        ha_close = (last['Open'] + last['High'] + last['Low'] + last['Close']) / 4
        # HA Open = (Prev_HA_Open + Prev_HA_Close)/2
        ha_open = (prev['Open'] + prev['Close']) / 2
        
        trend = "Bullish 🟢" if ha_close > ha_open else "Bearish 🔴"
        return trend, round(float(last['Close']), 2)
    except:
        return "ERROR", 0

# --- LOGIC 3: MULTI-FEED SENTIMENT ---
def get_crowd_sentiment(symbol):
    """Combines StockTwits retail bias and News headlines."""
    score = 0.0
    sources = []
    
    # Source A: StockTwits Cashtag (Direct Retail Sentiment)
    try:
        st_url = f"https://stocktwits.com{symbol}.json"
        res = requests.get(st_url, timeout=3).json()
        msgs = res.get('messages', [])
        if msgs:
            tags = [m['entities'].get('sentiment', {}).get('basic') for m in msgs]
            bull = tags.count('Bullish')
            bear = tags.count('Bearish')
            if (bull + bear) > 0:
                score += (bull - bear) / (bull + bear)
                sources.append(f"StockTwits (Bull:{bull}|Bear:{bear})")
    except: pass

    # Source B: News Feed
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        news = ticker.news[:3]
        if news:
            news_scores = [sia.polarity_scores(n['title'])['compound'] for n in news]
            score += sum(news_scores) / len(news_scores)
            sources.append("Yahoo Finance News")
    except: pass

    final_score = round(score / (len(sources) if sources else 1), 2)
    return final_score, " | ".join(sources) if sources else "No Crowd Data"

# --- UI & DEBUGGING STAGES ---
st.set_page_config(page_title="NSE F&O Debug Scanner", layout="wide")
st.title("🛡️ NSE F&O Scanner (Debug Mode)")

# Sidebar config
limit = st.sidebar.slider("Scan Count", 5, 50, 10)

if st.button("🚀 Start Live Debug Scan"):
    fno_list = get_fno_list()
    targets = fno_list[:limit]
    
    scan_results = []
    
    for s in targets:
        # VISUAL DEBUGGING BLOCKS
        with st.expander(f"🔍 Analyzing: {s}", expanded=True):
            col1, col2 = st.columns(2)
            
            # Stage 1: HA Trend
            with col1:
                st.write("**Stage 1: Heikin Ashi Logic**")
                trend, price = calculate_ha_trend(s)
                if trend != "ERROR":
                    st.success(f"Trend: {trend} | Price: ₹{price}")
                else:
                    st.error("Price fetch failed.")
            
            # Stage 2: Sentiment
            with col2:
                st.write("**Stage 2: Crowd Scoring**")
                sent_score, meta = get_crowd_sentiment(s)
                st.info(f"Score: {sent_score}")
                st.caption(f"Sources: {meta}")
            
            scan_results.append({
                "Symbol": s,
                "Price": price,
                "HA Trend": trend,
                "Crowd Score": sent_score,
                "Sources": meta
            })
            time.sleep(0.5) # Avoid API rate limits

    # FINAL CONSOLIDATED OUTPUT
    st.divider()
    st.subheader("📊 Final Market Dashboard")
    df = pd.DataFrame(scan_results)
    st.dataframe(df, use_container_width=True)
