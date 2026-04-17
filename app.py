import streamlit as st
import pandas as pd
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import time

# --- INITIALIZATION ---
@st.cache_resource
def setup_nlp():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = setup_nlp()

def get_fno_list():
    return ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS"]

# --- FIXED HEIKIN ASHI LOGIC ---
def calculate_ha_trend(symbol):
    try:
        # Fetch data and force single-level columns
        data = yf.download(symbol, period="5d", interval="1d", progress=False)
        
        # FIX: Check if empty using .empty correctly
        if data.empty or len(data) < 2:
            return "ERROR", 0, "No data available"

        # FIX: Flatten multi-index columns if they exist
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Access last two days safely
        today = data.iloc[-1]
        yesterday = data.iloc[-2]
        
        # HA Logic
        ha_close = (today['Open'] + today['High'] + today['Low'] + today['Close']) / 4
        ha_open = (yesterday['Open'] + yesterday['Close']) / 2
        
        trend = "Bullish 🟢" if ha_close > ha_open else "Bearish 🔴"
        return trend, round(float(today['Close']), 2), "Success"
    except Exception as e:
        return "ERROR", 0, str(e)

# --- FIXED SENTIMENT LOGIC ---
def get_crowd_score(symbol):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news or len(news) == 0:
            return 0.0, "No Headlines Found"
        
        # FIX: Access titles using .get() to prevent 'title' key errors
        scores = []
        for n in news[:5]:
            title = n.get('title', n.get('description', ''))
            if title:
                scores.append(sia.polarity_scores(title)['compound'])
        
        if not scores: return 0.0, "Zero scores generated"
        avg_score = round(sum(scores) / len(scores), 2)
        return avg_score, f"Analyzed {len(scores)} headlines"
    except Exception as e:
        return 0.0, f"Sentiment Error: {str(e)}"

# --- UI ---
st.title("🛡️ NSE F&O Debug Scanner (V2 Fixed)")

if st.button("🚀 Start Debug Scan"):
    stocks = get_fno_list()
    final_data = []
    
    for stock in stocks:
        with st.expander(f"🔍 Analyzing: {stock}", expanded=True):
            # Phase 1
            st.write("**Phase 1: Price Data**")
            trend, price, p_msg = calculate_ha_trend(stock)
            if trend != "ERROR":
                st.success(f"Trend: {trend} | Price: ₹{price}")
            else:
                st.error(f"Logic Error: {p_msg}")
            
            # Phase 2
            st.write("**Phase 2: Crowd Scoring**")
            score, s_msg = get_crowd_score(stock)
            st.info(f"Score: {score} | {s_msg}")

            final_data.append({"Stock": stock, "Price": price, "Trend": trend, "Sentiment": score})
            time.sleep(0.5)

    st.divider()
    st.dataframe(pd.DataFrame(final_data), use_container_width=True)
