import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

st.set_page_config(layout="wide")

st.title("🏹 NSE F&O Smoothed HA + SMC Scanner")


# =============================
# PARAMETERS
# =============================

SWING_LEN = 5
ATR_LEN = 3
ATR_THRESHOLD = 3.5
LEN1 = 5
LEN2 = 3


# =============================
# FETCH NSE F&O SYMBOLS (FAST API)
# =============================

@st.cache_data(ttl=86400)
def fetch_fno_symbols():

    url = "https://www.nseindia.com/api/market-data-pre-open?key=FO"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/"
    }

    session = requests.Session()

    session.get("https://www.nseindia.com", headers=headers)

    response = session.get(url, headers=headers)

    data = response.json()

    symbols = []

    for item in data["data"]:
        symbols.append(item["metadata"]["symbol"])

    return sorted(list(set(symbols)))


with st.spinner("Fetching NSE F&O universe..."):
    symbols = fetch_fno_symbols()

st.success(f"{len(symbols)} F&O symbols loaded")


# =============================
# DOWNLOAD OHLC DATA
# =============================

@st.cache_data
def download_data(symbols):

    tickers = [s + ".NS" for s in symbols]

    return yf.download(
        tickers=tickers,
        period="3mo",
        interval="1d",
        group_by="ticker",
        progress=False,
        threads=True
    )


with st.spinner("Downloading market data..."):
    data = download_data(symbols)

st.success("Market data ready")


# =============================
# HELPERS
# =============================

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def atr_percent(df):

    tr = pd.concat([
        df.High - df.Low,
        abs(df.High - df.Close.shift()),
        abs(df.Low - df.Close.shift())
    ], axis=1).max(axis=1)

    atr = tr.rolling(ATR_LEN).mean()

    return (atr / df.Close) * 100


def pivot_high(series, length):

    return series[
        (series.shift(length) < series)
        &
        (series.shift(-length) < series)
    ]


def pivot_low(series, length):

    return series[
        (series.shift(length) > series)
        &
        (series.shift(-length) > series)
    ]


# =============================
# START SCAN BUTTON
# =============================

if st.button("🚀 Start Scan"):

    progress = st.progress(0)
    status = st.empty()

    ce_candidates = []
    pe_candidates = []
    neutral = []

    total = len(symbols)


    for i, stock in enumerate(symbols):

        status.text(f"Scanning {stock} ({i+1}/{total})")

        ticker = stock + ".NS"

        if ticker not in data:
            continue

        df = data[ticker].dropna()

        if len(df) < 40:
            continue


        # ATR FILTER

        df["ATR%"] = atr_percent(df)

        if df["ATR%"].iloc[-1] < ATR_THRESHOLD:

            neutral.append(stock)
            continue


        # SMOOTHED HA

        sOpen = ema(df.Open, LEN1)
        sClose = ema(df.Close, LEN1)
        sHigh = ema(df.High, LEN1)
        sLow = ema(df.Low, LEN1)

        ha_close = (sOpen + sHigh + sLow + sClose) / 4

        ha_open = ha_close.copy()

        ha_open.iloc[0] = (sOpen.iloc[0] + sClose.iloc[0]) / 2

        for j in range(1, len(df)):

            ha_open.iloc[j] = (
                ha_open.iloc[j - 1]
                + ha_close.iloc[j - 1]
            ) / 2


        o2 = ema(ha_open, LEN2)
        c2 = ema(ha_close, LEN2)

        Hadiff = o2 - c2


        ha_buy = (

            df.Close.iloc[-1] > df.Open.iloc[-1]
            and Hadiff.iloc[-1] < 0
            and Hadiff.iloc[-4] > 0
            and df.Close.iloc[-1] > c2.iloc[-1]

        )


        ha_sell = (

            df.Close.iloc[-1] < df.Open.iloc[-1]
            and Hadiff.iloc[-1] > 0
            and Hadiff.iloc[-4] < 0
            and df.Close.iloc[-1] < c2.iloc[-1]

        )


        ph = pivot_high(df.High, SWING_LEN)
        pl = pivot_low(df.Low, SWING_LEN)


        lastH = ph.dropna().iloc[-1] if ph.dropna().size else np.nan
        lastL = pl.dropna().iloc[-1] if pl.dropna().size else np.nan


        bull_break = False
        bear_break = False


        if not np.isnan(lastH):
            bull_break = df.Close.iloc[-1] > lastH


        if not np.isnan(lastL):
            bear_break = df.Close.iloc[-1] < lastL


        buy_signal = bull_break or ha_buy
        sell_signal = bear_break or ha_sell


        if buy_signal:
            ce_candidates.append(stock)

        elif sell_signal:
            pe_candidates.append(stock)

        else:
            neutral.append(stock)


        progress.progress((i + 1) / total)


    status.text("Scan complete ✅")


    st.subheader("🟢 CE Candidates")
    st.write(ce_candidates)

    st.subheader("🔴 PE Candidates")
    st.write(pe_candidates)

    st.subheader("⚪ Neutral")
    st.write(neutral)
