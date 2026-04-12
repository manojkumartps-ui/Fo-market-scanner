import streamlit as st
import pandas as pd
import pandas_ta as ta
from growwapi import GrowwAPI
from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from datetime import datetime

# --- DASHBOARD UI ---
st.set_page_config(page_title="F&O AI Scanner", page_icon="📈")
st.title("🚀 F&O Market Scanner")

# State to keep track of scanning
if 'running' not in st.session_state:
    st.session_state.running = False
if 'top_picks' not in st.session_state:
    st.session_state.top_picks = []

# --- AI AGENT ---
agent = Agent(
    tools=[DuckDuckGoTools()],
    instructions=["Analyze opinions for the stock. Rank by sentiment score 1-10."]
)

# --- TECHNICAL LOGIC (SMOOTHED HA) ---
def is_logic_met(df):
    try:
        # Step 1 & 2: Smoothing and HA
        df['sC'] = ta.ema(df['Close'], length=5)
        df['sO'] = ta.ema(df['Open'], length=5)
        ha_c = (df['sO'] + df['High'] + df['Low'] + df['sC']) / 4
        # Step 3: Second Smoothing
        df['o2'] = ta.ema(df['sO'], length=3)
        df['c2'] = ta.ema(ha_c, length=3)
        df['diff'] = df['o2'] - df['c2']
        
        curr, prev3 = df.iloc[-1], df.iloc[-4]
        return (curr['Close'] > curr['Open']) and (curr['diff'] < 0) and (prev3['diff'] > 0)
    except: 
        return False

# --- SCANNER TRIGGER ---
col1, col2 = st.columns(2)
with col1:
    if st.button("▶ START SCANNING"):
        st.session_state.running = True

with col2:
    if st.button("⏹ STOP"):
        st.session_state.running = False
        st.warning("Scanner Stopped.")

# --- RESULTS DISPLAY ---
st.divider()
st.subheader("🏆 Top 3 Validated Candidates")

if st.session_state.running:
    now = datetime.now()
    is_weekend = now.weekday() >= 5
    market_open = now.replace(hour=9, minute=15, second=0)
    market_close = now.replace(hour=15, minute=30, second=0)

    # 1. TIME & DAY CHECKS
    if is_weekend:
        st.error("🚫 NO MARKET TODAY (Weekend)")
        st.session_state.running = False
    elif now < market_open or now > market_close:
        st.warning(f"🕒 OUTSIDE MARKET HOURS (9:15 - 15:30). Current: {now.strftime('%H:%M')}")
        st.session_state.running = False
    else:
        # 2. LIVE SCANNING LOGIC
        try:
            # Initialize Groww with secrets
            api = GrowwAPI(auth_token=st.secrets["GROWW_API_KEY"])
            st.info("🔍 Connecting to Groww API...")
            
            # This is where your ticker loop runs during market hours
            # For now, if no stocks match logic, we show this:
            if not st.session_state.top_picks:
                st.info("⌛ Scanning Stock Futures... NO CANDIDATES FOUND YET.")
            
            for pick in st.session_state.top_picks[:3]:
                st.metric(label=pick['ticker'], value=f"Score: {pick['score']}/10")
                st.write(f"**AI Verdict:** {pick['verdict']}")
                
        except Exception as e:
            # 3. API FAILURE CHECK
            st.error("⚠️ GROWW API NOT RESPONDING (Check Secrets or Connection)")
            st.session_state.running = False
else:
    st.info("Click 'Start' to begin real-time market analysis.")