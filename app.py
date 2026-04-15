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
        # Pulls the official NSE F&O underlying list
        url = "https://nsearchives.nseindia.com/content/fo/fo_underlyinglist.htm"
        header = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=header)
        df_list = pd.read_html(res.text)
        
        # The list is usually in the first table
        fno_df = df_list[0]
        fno_df.columns = fno_df.iloc[0] # Set header
        fno_df = fno_df[1:]
        
        # Format for yfinance (.NS suffix)
        tickers = fno_df['UNDERLYING'].unique().tolist()
        return [f"{str(t).strip()}.NS" for t in tickers if isinstance(t, str) and len(t) > 1]
    except Exception as e:
        st.error(f"Could not fetch live NSE list: {e}")
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"]

# --- 2. TECHNICAL LOGIC (Smoothed HA) ---
def check_logic(df):
    try:
        # Step 1: Initial Smoothing (5-period)
        df['sC'] = ta.ema(df['Close'], length=5)
        df['sO'] = ta.ema(df['Open'], length=5)
        
        # Step 2: Heikin-Ashi calculation
        ha_c = (df['sO'] + df['High'] + df['Low'] + df['sC']) / 4
        
        # Step 3: Second Smoothing (3-period)
        df['o2'] = ta.ema(df['sO'], length=3)
        df['c2'] = ta.ema(ha_c, length=3)
        df['diff'] = df['o2'] - df['c2']
        
        # Signal Points
        curr = df.iloc[-1]   # Today (D-1)
        prev1 = df.iloc[-2]  # Yesterday
        prev3 = df.iloc[-4]  # 3 days ago
        
        # Analysis
        is_green = curr['Close'] > curr['Open']
        
        # Strict Match (Original Logic: Flip from + to - over 3 days)
        strict_match = is_green and (curr['diff'] < 0) and (prev3['diff'] > 0)
        
        # Near Miss (For debugging: Flip happened in last 1-2 days)
        near_miss = is_green and (curr['diff'] < 0) and (prev1['diff'] > 0)
        
        return strict_match, near_miss
    except: 
        return False, False

# --- 3. AI AGENT SETUP ---
agent = Agent(
    tools=[DuckDuckGoTools()],
    instructions=[
        "Search for retail sentiment and crowd opinions on Indian finance forums, Twitter, and Reddit.",
        "Provide a 'Crowd Sentiment Score' (1-10) for a bullish move.",
        "Explain exactly what the 'crowd' is saying about this stock right now."
    ]
)

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="F&O AI Scanner", page_icon="📈", layout="wide")
st.title("🛡️ Automated D-1 F&O AI Scanner")
st.markdown("### Technical Logic: Smoothed HA Reversal | Sentiment: Crowd Pulse")

if st.button("🚀 RUN FULL MARKET SCAN"):
    with st.spinner("Fetching latest F&O list from NSE archives..."):
        fno_list = get_fno_tickers()
    
    st.write(f"🔍 Found {len(fno_list)} F&O stocks. Analyzing D-1 patterns...")
    
    results = []
    near_misses = []
    progress = st.progress(0)
    
    for i, ticker in enumerate(fno_list):
        # 60 days of D-1 data automatically fetched
        df = yf.download(ticker, period="60d", interval="1d", progress=False)
        
        if not df.empty and len(df) > 10:
            match, near = check_logic(df)
            
            if match:
                st.success(f"🎯 Perfect Match: {ticker}")
                with st.spinner(f"Agent scanning crowd sentiment for {ticker}..."):
                    response = agent.run(f"Current crowd sentiment and retail pulse for {ticker} stock in India.")
                    results.append({"ticker": ticker, "verdict": response.content})
            elif near:
                near_misses.append(ticker)
        
        progress.progress((i + 1) / len(fno_list))

    # --- DISPLAY RESULTS ---
    st.divider()
    if results:
        st.subheader("🏆 Validated Top Picks")
        for res in results:
            with st.expander(f"📈 {res['ticker']} - AI Analysis"):
                st.markdown(res['verdict'])
    
    if near_misses:
        st.subheader("⚠️ Near Misses (Debug)")
        st.info(f"The following stocks turned green and are flipping trend, but didn't meet the strict '3-day ago' rule: {', '.join(near_misses)}")

    if not results and not near_misses:
        st.warning("No candidates found. The current market trend might be too strong in one direction to trigger a reversal.")
