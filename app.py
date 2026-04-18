import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

st.set_page_config(layout="wide")
st.title("F&O Scanner — Latest Candle Signal Engine")

# ================= BULLET 1 ADOPTED =================
# Faster smoothing to catch Day 1 trend shifts
LEN1 = 3  
LEN2 = 2  


# ================= STABLE F&O SOURCE =================

@st.cache_data(ttl=86400)
def get_fno():
    """
    Fetches the actual F&O list from a reliable public CSV.
    This bypasses NSE blocking issues on Streamlit Cloud.
    """
    url = "https://kite.trade"
    try:
        df = pd.read_csv(url)
        # Filter for NSE Futures segment to get individual stocks
        fno_df = df[df['segment'] == 'NFO-FUT']
        return sorted(fno_df['name'].unique().tolist())
    except Exception as e:
        st.error(f"Critical Error: Unable to fetch stock list. {e}")
        return []

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
    # Standard HA calculation
    ha_close = (df.Open + df.High + df.Low + df.Close) / 4
    ha_open = np.zeros(len(df))
    # Corrected indexing for Streamlit compatibility
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
    if len(df) < 5: return "NEUTRAL", None

    o2, c2 = smoothed_ha(df)
    Hadiff = o2 - c2
    i = len(df) - 1

    # Signal Logic
    bullish = (Hadiff.iloc[i-1] <= 0 and Hadiff.iloc[i] > 0 and df.Close.iloc[i] > df.Open.iloc[i])
    bearish = (Hadiff.iloc[i-1] >= 0 and Hadiff.iloc[i] < 0 and df.Close.iloc[i] < df.Open.iloc[i])

    if bullish:
        return "BUY", {
            "hadiff_prev": float(Hadiff.iloc[i-1]),
            "hadiff_curr": float(Hadiff.iloc[i]),
            "close": float(df.Close.iloc[i]),
            "open": float(df.Open.iloc[i])
        }
    if bearish:
        return "SELL", {
            "hadiff_prev": float(Hadiff.iloc[i-1]),
            "hadiff_curr": float(Hadiff.iloc[i]),
            "close": float(df.Close.iloc[i]),
            "open": float(df.Open.iloc[i])
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
            buy_list.append(s)
            if buy_trace is None: buy_trace = {s: trace}
        elif signal == "SELL":
            sell_list.append(s)
            if sell_trace is None: sell_trace = {s: trace}
        else:
            neutral.append(s)

    st.subheader(f"🟢 BUY Candidates ({len(buy_list)})")
    st.write(buy_list)
    st.subheader(f"🔴 SELL Candidates ({len(sell_list)})")
    st.write(sell_list)
    st.divider()
    st.subheader("🧠 Example Trace")
    st.write({"BUY": buy_trace, "SELL": sell_trace})
