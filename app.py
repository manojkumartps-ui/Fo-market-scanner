import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

st.set_page_config(layout="wide")
st.title("SMC + Smoothed HA Scanner (TradingView-Parity Version)")


# ================= SETTINGS =================

LEN1 = 5
LEN2 = 3

ATR_LEN = 3
ATR_THRESHOLD = 3.5

GAP_MULT = 0.5
BOX_AGE_LIMIT = 15


# ================= TRUE TV EMA =================

def tv_ema(series, length):

    alpha = 2 / (length + 1)

    ema = series.copy()

    for i in range(1, len(series)):

        ema.iloc[i] = (
            alpha * series.iloc[i]
            + (1 - alpha) * ema.iloc[i - 1]
        )

    return ema


# ================= ATR =================

def atr(df):

    tr = pd.concat([
        df.High - df.Low,
        abs(df.High - df.Close.shift()),
        abs(df.Low - df.Close.shift())
    ], axis=1).max(axis=1)

    return tr.rolling(ATR_LEN).mean()


# ================= FETCH SYMBOLS =================

@st.cache_data(ttl=86400)
def fetch_symbols():

    session = requests.Session()

    headers = {"User-Agent": "Mozilla/5.0"}

    session.get("https://www.nseindia.com", headers=headers)

    url = "https://www.nseindia.com/api/market-data-pre-open?key=FO"

    data = session.get(url, headers=headers).json()

    return sorted([
        x["metadata"]["symbol"]
        for x in data["data"]
    ])


symbols = fetch_symbols()

st.success(f"{len(symbols)} symbols loaded")


# ================= DOWNLOAD DATA =================

@st.cache_data
def load_data(symbols):

    tickers = [x + ".NS" for x in symbols]

    return yf.download(
        tickers=tickers,
        period="6mo",
        interval="1d",
        group_by="ticker",
        threads=True
    )


data = load_data(symbols)

st.success("OHLC ready")


# ================= RUN SCAN =================

if st.button("RUN SCAN"):

    ce = []
    pe = []

    progress = st.progress(0)

    for i, symbol in enumerate(symbols):

        ticker = symbol + ".NS"

        if ticker not in data:
            continue

        df = data[ticker].dropna()

        if len(df) < 60:
            continue


        # ================= ATR FILTER =================

        df["ATR"] = atr(df)

        df["ATR%"] = df["ATR"] / df.Close * 100

        if df["ATR%"].iloc[-1] < ATR_THRESHOLD:
            continue


        # ================= TRUE TV SMOOTHED HA =================

        sOpen = tv_ema(df.Open.copy(), LEN1)
        sClose = tv_ema(df.Close.copy(), LEN1)
        sHigh = tv_ema(df.High.copy(), LEN1)
        sLow = tv_ema(df.Low.copy(), LEN1)


        ha_close = (sOpen + sHigh + sLow + sClose) / 4


        ha_open = ha_close.copy()

        ha_open.iloc[0] = (sOpen.iloc[0] + sClose.iloc[0]) / 2


        for j in range(1, len(df)):

            ha_open.iloc[j] = (
                ha_open.iloc[j - 1]
                + ha_close.iloc[j - 1]
            ) / 2


        o2 = tv_ema(ha_open.copy(), LEN2)

        c2 = tv_ema(ha_close.copy(), LEN2)


        Hadiff = o2 - c2


        # ================= HA SIGNAL =================

        ha_buy = (

            df.Close.iloc[-1] > df.Open.iloc[-1]
            and Hadiff.iloc[-1] < 0
            and max(Hadiff.iloc[-4:-1]) > 0
            and df.Close.iloc[-1] > c2.iloc[-1]

        )


        ha_sell = (

            df.Close.iloc[-1] < df.Open.iloc[-1]
            and Hadiff.iloc[-1] > 0
            and min(Hadiff.iloc[-4:-1]) < 0
            and df.Close.iloc[-1] < c2.iloc[-1]

        )


        if ha_buy:
            ce.append(symbol)

        if ha_sell:
            pe.append(symbol)


        progress.progress((i + 1) / len(symbols))


    st.success("Scan complete")


    st.subheader("CE Candidates")
    st.write(ce)


    st.subheader("PE Candidates")
    st.write(pe)
