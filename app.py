import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
import time

# --- 1. TECHNICAL LOGIC (Smoothed HA) ---
def is_logic_met(df):
    try:
        # Step 1: Initial Smoothing
        df['sC'] = ta.ema(df['Close'], length=5)
        df['sO'] = ta.ema(df['Open'], length=5)
        
        # Step 2: Heikin-Ashi calculation
        ha_c = (df['sO'] + df['High'] + df['Low'] + df['sC']) / 4
        
        # Step 3: Second Smoothing
        df['o2'] = ta.ema(df['sO'], length=3)
        df['c2'] = ta.ema(ha_c, length=3)
        df['diff'] = df['o2'] - df['c2']
        
        # Get D-1 (yesterday) and the day from 3 bars ago
        curr = df.iloc[-1]
        prev3 = df.iloc[-4]
        
        # Logic: Bullish Candle + Trend Reversal (diff flips from + to -)
        return (curr['Close'] > curr['Open']) and (curr['diff'] < 0) and (prev3['diff'] > 0)
    except Exception: 
        return False

# --- 2. SENTIMENT AGENT ---
agent = Agent(
    tools=[DuckDuckGoTools()],
    instructions=[
        "Search for retail sentiment and crowd opinions on Twitter, Reddit, and news sites.",
        "Provide a 'Crowd Sentiment Score' (1-10) and a brief justification.",
        "Highlight any major news that justifies why the crowd is bullish/bearish today."
    ]
)

# --- 3. UI SETUP ---
st.set_page_config(page_title="F&O AI Scanner", page_icon="📈", layout="wide")
st.title("🛡️ Automated D-1 F&O AI Scanner")
st.markdown("### Technical Logic: Smoothed HA Reversal | Sentiment: Crowd Pulse")

# The complete list of 180+ F&O stocks (abbreviated here for brevity)
FNO_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "SBIN.NS", 
    "BHARTIARTL.NS", "LICI.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS", "BAJFINANCE.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "TATASTEEL.NS",
    "JSWSTEEL.NS", "POWERGRID.NS", "NTPC.NS", "M&M.NS", "HCLTECH.NS", "ONGC.NS"
    # ... You can paste the remaining NSE F&O tickers here ...
]

if st.button("🚀 RUN SCANNER"):
    st.session_state.results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(FNO_TICKERS):
        status_text.text(f"Scanning {ticker} ({i+1}/{len(FNO_TICKERS)})")
        
        # TOUCH-FREE DATA: Automatically pulls 60 days of D-1 data
        df = yf.download(ticker, period="60d", interval="1d", progress=False)
        
        if is_logic_met(df):
            st.success(f"🎯 Pattern Match: {ticker}")
            # Run the AI Crowd Sentiment Agent
            with st.spinner("Analyzing Crowd Sentiment..."):
                response = agent.run(f"Latest news and retail sentiment for {ticker} stock in India.")
                st.session_state.results.append({"ticker": ticker, "verdict": response.content})
        
        progress_bar.progress((i + 1) / len(FNO_TICKERS))
    
    st.divider()
    if st.session_state.results:
        for res in st.session_state.results:
            with st.expander(f"📈 {res['ticker']} - AI Analysis"):
                st.markdown(res['verdict'])
    else:
        st.info("No candidates found meeting the reversal criteria for D-1.")
