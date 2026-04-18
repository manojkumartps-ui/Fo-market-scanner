import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

st.set_page_config(layout="wide")
st.title("F&O Scanner — High-Speed Signal Engine (Bullet 1 Adopted)")

# ================= BULLET 1 ADJUSTMENTS =================
# Reduced smoothing to decrease lag and increase signal frequency
LEN1 = 3  
LEN2 = 2


# ================= F&O LIST =================

@st.cache_data(ttl=86400)
def get_fno():
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}
    session.get("https://nseindia.com", headers=headers)
    url = "https://nseindia.com/api/market-data-pre-open?key=FO"
    data = session.get(url, headers=headers).json()
    return sorted(set(x["metadata"]["symbol"] for x in data["data"]))

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
    ha_open[0] = (df.Open.iloc[0] + df.Close.iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
    return pd.Series(ha_open, index=df.index), pd.Series(ha_close, index=df.index)


# ================= SMOOTHED HA =================

def smoothed_ha(df):
    ha_open, ha_close = heikin_ashi(df)
    # Using the faster LEN1 and LEN2 settings here
    o1 = ha_open.ewm(span=LEN1, adjust=False).mean()
    c1 = ha_close.ewm(span=LEN1, adjust=False).mean()
    o2 = o1.ewm(span=LEN2, adjust=False).mean()
    c2 = c1.ewm(span=LEN2, adjust=False).mean()
    return o2, c2


# ================= LATEST CANDLE ENGINE =================

def evaluate_latest(df):
    df = df.dropna().copy()
    if len(df) < 5: return "NEUTRAL", None

    o2, c2 = smoothed_ha(df)
    Hadiff = o2 - c2
    i = len(df) - 1

    bullish = (
        Hadiff.iloc[i-1] <= 0 and
        Hadiff.iloc[i] > 0 and
        df.Close.iloc[i] > df.Open.iloc[i]
    )

    bearish = (
        Hadiff.iloc[i-1] >= 0 and
        Hadiff.iloc[i] < 0 and
        df.Close.iloc[i] < df.Open.iloc[i]
    )

    if bullish:
        return "BUY", {"hadiff_prev": float(Hadiff.iloc[i-1]), "hadiff_curr": float(Hadiff.iloc[i]), "close": float(df.Close.iloc[i]), "open": float(df.Open.iloc[i])}
    if bearish:
        return "SELL", {"hadiff_prev": float(Hadiff.iloc[i-1]), "hadiff_curr": float(Hadiff.iloc[i]), "close": float(df.Close.iloc[i]), "open": float(df.Open.iloc[i])}
    
    return "NEUTRAL", None


# ================= RUN =================

if st.button("RUN SCAN"):
    buy_list = []
    sell_list = []
    neutral = []
    buy_trace = None
    sell_trace = None

    for s in symbols:
        ticker = s + ".NS"
        if ticker not in data or data[ticker].empty:
            continue
        df = data[ticker]
        signal, trace = evaluate_latest(df)
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
    st.subheader("🧠 Example Trace (BUY)")
    st.write(buy_trace)
    st.subheader("🧠 Example Trace (SELL)")
    st.write(sell_trace)
