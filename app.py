import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(layout="wide")
st.title("F&O Scanner — Momentum + SMC Engine")

# ================= SETTINGS =================
LEN1 = 3  
LEN2 = 2  
SMC_LOOKBACK = 10

# ================= BULLETPROOF F&O SOURCE =================

@st.cache_data(ttl=86400)
def get_fno():
    """
    Fetches F&O list from a static GitHub source to avoid 403 Forbidden errors.
    """
    # Using a reliable community-maintained CSV of NSE F&O stocks
    url = "https://githubusercontent.com"
    try:
        df = pd.read_csv(url)
        return sorted(df['Symbol'].unique().tolist())
    except Exception as e:
        # Emergency hardcoded list if all network sources fail
        st.warning("Fetching from primary source failed, using internal list.")
        return ["MARICO", "SUPREMEIND", "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "BHARTIARTL", "AXISBANK"]

symbols = get_fno()

# ================= DATA =================

@st.cache_data
def load(symbols):
    if not symbols:
        return pd.DataFrame()
    tickers = [s + ".NS" for s in symbols]
    return yf.download(
        tickers=tickers,
        period="6mo",
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False
    )

data = load(symbols)

# ================= HEIKIN ASHI =================

def heikin_ashi(df):
    ha_close = (df.Open + df.High + df.Low + df.Close) / 4
    ha_open = np.zeros(len(df))
    ha_open[0] = (df.Open.iloc[0] + df.Close.iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
    return pd.Series(ha_open, index=df.index), pd.Series(ha_close, index=df.index)

# ================= SMOOTHED HA =================

def smoothed_ha(df):
    ha_open, ha_close = heikin_ashi(df)
    o1 = ha_open.ewm(span=LEN1, adjust=False).mean()
    c1 = ha_close.ewm(span=LEN1, adjust=False).mean()
    o2 = o1.ewm(span=LEN2, adjust=False).mean()
    c2 = c1.ewm(span=LEN2, adjust=False).mean()
    return o2, c2

# ================= SIGNAL ENGINE =================

def evaluate_latest(df):
    df = df.dropna().copy()
    if len(df) < 20: return "NEUTRAL", None

    o2, c2 = smoothed_ha(df)
    Hadiff = o2 - c2
    i = len(df) - 1

    # --- YOUR ORIGINAL LOGIC (UNTOUCHED) ---
    bullish = (Hadiff.iloc[i-1] <= 0 and Hadiff.iloc[i] > 0 and df.Close.iloc[i] > df.Open.iloc[i])
    bearish = (Hadiff.iloc[i-1] >= 0 and Hadiff.iloc[i] < 0 and df.Close.iloc[i] < df.Open.iloc[i])

    # --- ADDITIONAL SMC LOGIC ---
    swing_high = df['High'].iloc[-SMC_LOOKBACK:-1].max()
    swing_low = df['Low'].iloc[-SMC_LOOKBACK:-1].min()
    
    smc_bullish = (df.Close.iloc[i] > swing_high and c2.iloc[i] > o2.iloc[i])
    smc_bearish = (df.Close.iloc[i] < swing_low and c2.iloc[i] < o2.iloc[i])

    if bullish or smc_bullish:
        return "BUY", {"type": "Momentum" if bullish else "SMC BOS", "price": float(df.Close.iloc[i])}
    if bearish or smc_bearish:
        return "SELL", {"type": "Momentum" if bearish else "SMC BOS", "price": float(df.Close.iloc[i])}
    
    return "NEUTRAL", None

# ================= RUN =================

if st.button("RUN SCAN"):
    buy_list, sell_list = [], []
    
    for s in symbols:
        ticker = s + ".NS"
        if ticker not in data or data[ticker].empty: continue
        signal, trace = evaluate_latest(data[ticker])
        if signal == "BUY":
            buy_list.append(f"{s} ({trace['type']})")
        elif signal == "SELL":
            sell_list.append(f"{s} ({trace['type']})")

    st.subheader(f"🟢 BUY Candidates ({len(buy_list)})")
    st.write(buy_list)
    st.subheader(f"🔴 SELL Candidates ({len(sell_list)})")
    st.write(sell_list)
