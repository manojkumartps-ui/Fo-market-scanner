import streamlit as st
import pandas as pd
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import time

# --- STAGE 0: INITIALIZATION ---
@st.cache_resource
def setup_nlp():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = setup_nlp()

def get_fno_list():
    return ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS"]

# --- STAGE 1: HEIKIN ASHI WITH STEP-BY-STEP DEBUG ---
def calculate_ha_trend_debug(symbol):
    try:
        # FORCE: multi_level_index=False ensures a flat table
        df = yf.download(symbol, period="5d", interval="1d", progress=False, multi_level_index=False)
        
        if df.empty or len(df) < 2:
            return "ERROR", 0, "Insufficient Data (Need 2+ days)"

        # DEBUG: Show the raw data we received
        st.write(f"   📊 *DEBUG: Raw Data Sample for {symbol}*")
        st.dataframe(df.tail(2))

        # Standard OHLC
        today = df.iloc[-1]
        prev = df.iloc[-2]
        
        # STEP 1: HA Close = (O+H+L+C)/4
        ha_close = (today['Open'] + today['High'] + today['Low'] + today['Close']) / 4
        
        # STEP 2: HA Open = (Prev_Open + Prev_Close) / 2
        ha_open = (prev['Open'] + prev['Close']) / 2
        
        st.write(f"   📐 *DEBUG: HA Calculation -> Open: {round(ha_open, 2)} | Close: {round(ha_close, 2)}*")
        
        trend = "Bullish 🟢" if ha_close > ha_open else "Bearish 🔴"
        return trend, round(float(today['Close']), 2), "Success"
    except Exception as e:
        return "ERROR", 0, f"HA Logic Failed: {str(e)}"

# --- STAGE 2: SENTIMENT WITH DATA INSPECTION ---
def get_crowd_score_debug(symbol):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        if not news:
            return 0.0, "No Headlines Found"

        st.write(f"   📝 *DEBUG: Found {len(news)} headlines. Scoring...*")
        
        scores = []
        for n in news[:5]:
            # Inspect structure for titles or descriptions
            text = n.get('title') or n.get('description')
            if text:
                val = sia.polarity_scores(text)['compound']
                scores.append(val)
                st.write(f"      - {text[:50]}... | Score: {val}")

        if not scores: return 0.0, "Zero valid scores"
        avg_score = round(sum(scores) / len(scores), 2)
        return avg_score, "Calculated Successfully"
    except Exception as e:
        return 0.0, f"Sentiment Debug Error: {str(e)}"

# --- UI ---
st.title("🧪 Full-Logic NSE Debug Scanner")

if st.button("🚀 Start Deep-Logic Scan"):
    stocks = get_fno_list()
    final_results = []
    
    for stock in stocks:
        with st.expander(f"🔍 DEEP DEBUG: {stock}", expanded=True):
            # Phase 1: HA
            st.markdown("### 🟢 Phase 1: Heikin Ashi Logic")
            trend, price, p_msg = calculate_ha_trend_debug(stock)
            if trend != "ERROR":
                st.success(f"Final Trend: {trend}")
            else:
                st.error(p_msg)
            
            # Phase 2: Sentiment
            st.markdown("### 🔵 Phase 2: Crowd Scoring")
            score, s_msg = get_crowd_score_debug(stock)
            st.info(f"Final Score: {score} ({s_msg})")

            final_results.append({"Stock": stock, "Price": price, "Trend": trend, "Sentiment": score})
            time.sleep(1)

    st.divider()
    st.subheader("📊 Final Market Dashboard")
    st.dataframe(pd.DataFrame(final_results), use_container_width=True)
