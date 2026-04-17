import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

st.set_page_config(layout="wide")
st.title("NSE F&O Strategy Engine (Clean State Machine Build)")


# ================= SETTINGS =================

LEN1 = 5
LEN2 = 3
ATR_LEN = 3
ATR_THRESHOLD = 3.5

DEBUG = st.checkbox("Debug Mode")


# ================= F&O SYMBOLS =================

@st.cache_data(ttl=86400)
def get_fno_symbols():

    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    session.get("https://www.nseindia.com", headers=headers)

    url = "https://www.nseindia.com/api/market-data-pre-open?key=FO"
    data = session.get(url, headers=headers).json()

    return sorted(set(
        x["metadata"]["symbol"]
        for x in data["data"]
    ))


symbols = get_fno_symbols()
st.success(f"Loaded F&O symbols: {len(symbols)}")


# ================= DATA =================

@st.cache_data
def load_data(symbols):

    tickers = [s + ".NS" for s in symbols]

    return yf.download(
        tickers=tickers,
        period="6mo",
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False
    )


data = load_data(symbols)
st.success("OHLC loaded")


# ================= ATR =================

def atr(df):

    tr = pd.concat([
        df.High - df.Low,
        abs(df.High - df.Close.shift()),
        abs(df.Low - df.Close.shift())
    ], axis=1).max(axis=1)

    return tr.rolling(ATR_LEN).mean()


# ================= HEIKIN ASHI =================

def heikin_ashi(df):

    ha_close = (df.Open + df.High + df.Low + df.Close) / 4

    ha_open = np.zeros(len(df))
    ha_open[0] = (df.Open.iloc[0] + df.Close.iloc[0]) / 2

    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2

    return ha_open, ha_close


# ================= ENGINE =================

def run_strategy(df):

    df = df.dropna().copy()
    if len(df) < 60:
        return "NONE"


    # ATR
    df["ATR"] = atr(df)


    # HA
    ha_open, ha_close = heikin_ashi(df)


    # Smoothed HA
    o1 = pd.Series(ha_open).ewm(span=LEN1, adjust=False).mean()
    c1 = pd.Series(ha_close).ewm(span=LEN1, adjust=False).mean()

    o2 = o1.ewm(span=LEN2, adjust=False).mean()
    c2 = c1.ewm(span=LEN2, adjust=False).mean()

    Hadiff = o2 - c2


    state = "NONE"


    # ================= STATE MACHINE =================

    for i in range(5, len(df)):

        atr_ok = (df["ATR"].iloc[i] / df.Close.iloc[i]) * 100 >= ATR_THRESHOLD

        if not atr_ok:
            continue


        bullish_flip = (
            Hadiff.iloc[i] < 0 and
            Hadiff.iloc[i-1] > 0 and
            df.Close.iloc[i] > df.Open.iloc[i]
        )

        bearish_flip = (
            Hadiff.iloc[i] > 0 and
            Hadiff.iloc[i-1] < 0 and
            df.Close.iloc[i] < df.Open.iloc[i]
        )


        if bullish_flip:
            state = "CE"

        elif bearish_flip:
            state = "PE"


    return state


# ================= RUN =================

if st.button("RUN SCAN"):

    ce = []
    pe = []
    neutral = []

    progress = st.progress(0)

    total = len(symbols)


    for idx, s in enumerate(symbols):

        ticker = s + ".NS"

        if ticker not in data:
            continue

        df = data[ticker]


        state = run_strategy(df)


        if state == "CE":
            ce.append(s)

        elif state == "PE":
            pe.append(s)

        else:
            neutral.append(s)


        if DEBUG:
            st.write(s, state)


        progress.progress((idx+1)/total)


    st.success("SCAN COMPLETE")


    st.subheader("🟢 CE Candidates")
    st.write(ce)

    st.subheader("🔴 PE Candidates")
    st.write(pe)

    st.subheader("⚪ Neutral")
    st.write(neutral)
