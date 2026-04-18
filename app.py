import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

st.set_page_config(layout="wide")
st.title("F&O Scanner — Momentum + SMC Engine")

LEN1 = 3
LEN2 = 2
SMC_LOOKBACK = 10 # Lookback for BOS/CHoCH

# ================= F&O LIST =================

@st.cache_data(ttl=86400)
def get_fno():
    # Stable source to avoid JSONDecodeError on Streamlit Cloud
    url = "https://kite.trade"
    try:
        df = pd.read_csv(url)
        return sorted(df[df['segment'] == 'NFO-FUT']['name'].unique().tolist())
    except:
        return ["MARICO", "SUPREMEIND", "RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"]

symbols = get_fno()

# ================= DATA =================

@st.cache_data
def load(symbols):
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
    ha_open = (df.Open.iloc + df.Close.iloc) / 2
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

# ================= LATEST CANDLE ENGINE + SMC =================

def evaluate_latest(df):
    df = df.dropna().copy()
    if len(df) < 20: return "NEUTRAL", None

    o2, c2 = smoothed_ha(df)
    Hadiff = o2 - c2
    i = len(df) - 1

    # --- YOUR ORIGINAL LOGIC (UNTOUCHED) ---
    sha_bullish = (Hadiff.iloc[i-1] <= 0 and Hadiff.iloc[i] > 0 and df.Close.iloc[i] > df.Open.iloc[i])
    sha_bearish = (Hadiff.iloc[i-1] >= 0 and Hadiff.iloc[i] < 0 and df.Close.iloc[i] < df.Open.iloc[i])

    # --- ADDITIONAL SMC LOGIC (BOS/CHoCH) ---
    # Break of Structure: Price breaks the highest/lowest of the recent lookback
    swing_high = df['High'].iloc[-SMC_LOOKBACK:-1].max()
    swing_low = df['Low'].iloc[-SMC_LOOKBACK:-1].min()
    
    smc_bullish = (df.Close.iloc[i] > swing_high and c2.iloc[i] > o2.iloc[i])
    smc_bearish = (df.Close.iloc[i] < swing_low and c2.iloc[i] < o2.iloc[i])

    if sha_bullish or smc_bullish:
        return "BUY", {
            "signal_type": "SHA Crossover" if sha_bullish else "SMC BOS",
            "hadiff_curr": float(Hadiff.iloc[i]),
            "close": float(df.Close.iloc[i])
        }

    if sha_bearish or smc_bearish:
        return "SELL", {
            "signal_type": "SHA Crossover" if sha_bearish else "SMC BOS",
            "hadiff_curr": float(Hadiff.iloc[i]),
            "close": float(df.Close.iloc[i])
        }

    return "NEUTRAL", None

# ================= RUN =================

if st.button("RUN SCAN"):
    buy_list, sell_list, neutral = [], [], []
    buy_trace, sell_trace = None, None

    for s in symbols:
        ticker = s + ".NS"
        if ticker not in data or data[ticker].empty: continue
        signal, trace = evaluate_latest(data[ticker])
        
        if signal == "BUY":
            buy_list.append(f"{s} ({trace['signal_type']})")
            if buy_trace is None: buy_trace = {s: trace}
        elif signal == "SELL":
            sell_list.append(f"{s} ({trace['signal_type']})")
            if sell_trace is None: sell_trace = {s: trace}
        else:
            neutral.append(s)

    st.subheader("🟢 BUY Candidates")
    st.write(buy_list)
    st.subheader("🔴 SELL Candidates")
    st.write(sell_list)
    st.divider()
    st.subheader("🧠 Example Trace")
    st.write({"Latest Buy": buy_trace, "Latest Sell": sell_trace})
