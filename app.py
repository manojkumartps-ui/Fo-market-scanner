import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
import requests

# --- 1. PROGRAMMATIC F&O TICKER FETCH ---
def get_fno_tickers():
    try:
        # Fetching the F&O list from a reliable public source
        url = "https://www.niftytrader.in/nse-fo-lot-size"
        header = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=header)
        df_list = pd.read_html(res.text)
        
        # Typically the first table on this page contains the F&O symbols
        fno_df = df_list[0]
        # Clean symbols and add .NS for yfinance
        tickers = fno_df['SYMBOL'].unique().tolist()
        return [f"{t.strip()}.NS" for t in tickers if isinstance(t, str)]
    except Exception as e:
        # Fallback if scraping fails
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS"]

# --- 2. TECHNICAL LOGIC (Smoothed HA) ---
def is_logic_met(df):
    try:
        # 60 days ensures these EMAs are mathematically stable
        df['sC'] = ta.ema(df['Close'], length=5)
        df['sO'] = ta.ema(df['Open'], length=5)
        ha_c = (df['sO'] + df['High'] + df['Low'] + df['sC']) / 4
        
        df['o2'] = ta.ema(df['sO'], length=3)
        df['c2'] = ta.ema(ha_c, length=3)
        df['diff'] = df['o2'] - df['c2']
        
        curr, prev3 = df.iloc[-1], df.iloc[-4]
        return (curr['Close'] > curr['Open']) and (curr['diff'] < 0) and (prev3['diff'] > 0)
    except: 
        return False

# --- 3. AI CROWD SENTIMENT AGENT ---
agent = Agent(
    tools=[DuckDuckGoTools()],
    instructions=[
        "Search for retail sentiment and crowd opinions on Indian finance forums, Twitter, and Reddit.",
        "Provide a 'Crowd Confidence Score' (1-10) for a bullish move.",
        "Summarize why the crowd is currently interested in this stock."
    ]
)

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="F&O AI Scanner", layout="wide")
st.title("🚀 100% Touch-Free F&O AI Predictor")

if st.button("▶ START SCAN"):
    with st.spinner("Fetching live F&O list from NSE sources..."):
        fno_list = get_fno_tickers()
    
    st.write(f"🔍 Found {len(fno_list)} F&O stocks. Starting technical scan...")
    
    results = []
    progress = st.progress(0)
    
    for i, ticker in enumerate(fno_list):
        # Automatically pulls 60 days of D-1 data
        df = yf.download(ticker, period="60d", interval="1d", progress=False)
        
        if is_logic_met(df):
            st.success(f"✅ Technical Pattern Match: {ticker}")
            # AI Crowd Analysis
            with st.spinner(f"Analyzing Crowd Sentiment for {ticker}..."):
                response = agent.run(f"Current crowd sentiment and retail pulse for {ticker} stock in India.")
                results.append({"ticker": ticker, "verdict": response.content})
        
        progress.progress((i + 1) / len(fno_list))

    st.divider()
    if results:
        for res in results:
            with st.expander(f"📈 {res['ticker']} - Crowd Sentiment Analysis"):
                st.markdown(res['verdict'])
    else:
        st.info("No candidates found meeting technical reversal criteria today.")
