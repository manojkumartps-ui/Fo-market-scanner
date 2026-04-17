import streamlit as st
import pandas as pd
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import time

# --- STAGE 0: INITIALIZATION ---
@st.cache_resource
def setup_nlp():
    """Download lexicon and setup sentiment analyzer."""
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = setup_nlp()

# --- STAGE 1: LIST BUILDING ---
def get_fno_list():
    """
    Builds a reliable list of major NSE F&O stocks.
    Using .NS suffix for Yahoo Finance compatibility.
    """
    return [
        "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", 
        "SBIN.NS", "BHARTIARTL.NS", "AXISBANK.NS", "ITC.NS", "LT.NS",
        "KOTAKBANK.NS", "HINDUNILVR.NS", "M&M.NS", "BAJFINANCE.NS", "ADANIENT.NS"
    ]

# --- STAGE 2: HEIKIN ASHI LOGIC ---
def calculate_ha_trend(symbol):
    """
    Fetches OHLC and returns Heikin Ashi trend.
    HA Close = (Open + High + Low + Close) / 4
    HA Open = (Prev_Open + Prev_Close) / 2
    """
    try:
        # period="2d" is mandatory to get previous day for HA_Open
        data = yf.download(symbol, period="2d", interval="1d", progress=False)
        
        if data.empty or len(data) < 2:
            return "ERROR", 0, "Insufficient Data"
        
        # Latest bar (Today)
        today = data.iloc[-1]
        # Previous bar (Yesterday)
        yesterday = data.iloc[-2]
        
        # Calculation
        ha_close = (today['Open'] + today['High'] + today['Low'] + today['Close']) / 4
        ha_open = (yesterday['Open'] + yesterday['Close']) / 2
        
        trend = "Bullish 🟢" if ha_close > ha_open else "Bearish 🔴"
        return trend, round(float(today['Close']), 2), "Success"
    except Exception as e:
        return "ERROR", 0, str(e)

# --- STAGE 3: CROWD SENTIMENT ---
def get_crowd_score(symbol):
    """Fetches news via yfinance and calculates sentiment score."""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news:
            return 0.0, "No Headlines Found"
        
        # Extract and score headlines
        headlines = [n['title'] for n in news[:5]]
        scores = [sia.polarity_scores(h)['compound'] for h in headlines]
        avg_score = round(sum(scores) / len(scores), 2)
        
        return avg_score, f"Analyzed {len(headlines)} headlines"
    except Exception as e:
        return 0.0, f"Error: {str(e)}"

# --- MAIN DASHBOARD & DEBUGGING ---
st.set_page_config(page_title="NSE F&O Scanner", layout="wide")
st.title("🛡️ NSE F&O Trend & Sentiment Debugger")

# Sidebar
scan_count = st.sidebar.slider("Number of Stocks to Scan", 5, 15, 5)

if st.button("🚀 Start Live Market Debug"):
    fno_stocks = get_fno_list()
    targets = fno_stocks[:scan_count]
    
    final_data = []
    
    for stock in targets:
        # VISUAL DEBUGGING START
        with st.expander(f"🔍 Analyzing: {stock}", expanded=True):
            st.write(f"### Evaluating {stock}...")
            
            # --- PHASE 1: PRICE DATA ---
            st.write("**Phase 1: Heikin Ashi Calculation**")
            trend, price, p_msg = calculate_ha_trend(stock)
            if trend != "ERROR":
                st.success(f"✅ Data Received. HA Trend: {trend} | Price: ₹{price}")
            else:
                st.error(f"❌ Data Failed: {p_msg}")
            
            # --- PHASE 2: SENTIMENT DATA ---
            st.write("**Phase 2: Crowd Scoring**")
            score, s_msg = get_crowd_score(stock)
            if "Analyzed" in s_msg:
                st.info(f"✅ Sentiment: {score} ({s_msg})")
            else:
                st.warning(f"⚠️ Sentiment Issue: {s_msg}")

            # Collect results
            final_data.append({
                "Stock": stock.replace(".NS", ""),
                "Price": price,
                "HA Trend": trend,
                "Crowd Score": score,
                "Debug Info": s_msg
            })
            
            # Respect API limits
            time.sleep(0.5)

    # FINAL OUTPUT TABLE
    st.divider()
    st.subheader("📊 Final Evaluation Summary")
    if final_data:
        df = pd.DataFrame(final_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.error("No data collected during scan.")

st.caption("Using official yfinance data streams. No scraping required.")
