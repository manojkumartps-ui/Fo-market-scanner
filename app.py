import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

st.set_page_config(layout="wide")
st.title("📊 NSE F&O SMC + Smoothed HA Scanner (Rule-4 Corrected)")

# ================= PARAMETERS =================

SWING_LEN = 5
ATR_LEN = 3
ATR_THRESHOLD = 3.5
GAP_MULT = 0.5
BOX_AGE_LIMIT = 15

LEN1 = 5
LEN2 = 3


# ================= FETCH F&O SYMBOLS =================

@st.cache_data(ttl=86400)
def fetch_symbols():

    session = requests.Session()

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    session.get("https://www.nseindia.com", headers=headers)

    url = "https://www.nseindia.com/api/market-data-pre-open?key=FO"

    response = session.get(url, headers=headers)

    data = response.json()

    symbols = []

    for item in data["data"]:
        symbols.append(item["metadata"]["symbol"])

    return sorted(list(set(symbols)))


symbols = fetch_symbols()

st.success(f"{len(symbols)} symbols loaded")


# ================= DOWNLOAD DATA =================

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

st.success("OHLC ready")


# ================= HELPERS =================

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def atr(df):

    tr = pd.concat([
        df.High - df.Low,
        abs(df.High - df.Close.shift()),
        abs(df.Low - df.Close.shift())
    ], axis=1).max(axis=1)

    return tr.rolling(ATR_LEN).mean()


def pivot_high(series):

    return series[
        (series.shift(SWING_LEN) < series)
        &
        (series.shift(-SWING_LEN) < series)
    ]


def pivot_low(series):

    return series[
        (series.shift(SWING_LEN) > series)
        &
        (series.shift(-SWING_LEN) > series)
    ]


# ================= SCANNER =================

if st.button("🚀 RUN SCAN"):

    progress = st.progress(0)

    ce = []
    pe = []
    neutral = []

    total = len(symbols)


    for i, stock in enumerate(symbols):

        ticker = stock + ".NS"

        if ticker not in data:
            continue

        df = data[ticker].dropna()

        if len(df) < 60:
            continue


        # ================= ATR FILTER =================

        df["ATR"] = atr(df)
        df["ATR%"] = df["ATR"] / df.Close * 100

        if df["ATR%"].iloc[-1] < ATR_THRESHOLD:

            neutral.append(stock)
            continue


        # ================= SMOOTHED HA =================

        sOpen = ema(df.Open, LEN1)
        sClose = ema(df.Close, LEN1)
        sHigh = ema(df.High, LEN1)
        sLow = ema(df.Low, LEN1)

        ha_close = (sOpen + sHigh + sLow + sClose) / 4

        ha_open = ha_close.copy()

        ha_open.iloc[0] = (sOpen.iloc[0] + sClose.iloc[0]) / 2

        for j in range(1, len(df)):

            ha_open.iloc[j] = (
                ha_open.iloc[j-1]
                + ha_close.iloc[j-1]
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


        ha_buy_prev = (

            df.Close.iloc[-2] > df.Open.iloc[-2]
            and Hadiff.iloc[-2] < 0
            and Hadiff.iloc[-5] > 0
            and df.Close.iloc[-2] > c2.iloc[-2]

        )


        ha_sell = (

            df.Close.iloc[-1] < df.Open.iloc[-1]
            and Hadiff.iloc[-1] > 0
            and Hadiff.iloc[-4] < 0
            and df.Close.iloc[-1] < c2.iloc[-1]

        )


        ha_sell_prev = (

            df.Close.iloc[-2] < df.Open.iloc[-2]
            and Hadiff.iloc[-2] > 0
            and Hadiff.iloc[-5] < 0
            and df.Close.iloc[-2] < c2.iloc[-2]

        )


        # ================= FVG DETECTION =================

        last_bull_fvg_top = None
        last_bull_bar = None

        last_bear_fvg_bot = None
        last_bear_bar = None


        for k in range(2, len(df)):

            gap = df["ATR"].iloc[k] * GAP_MULT


            if df.Low.iloc[k] > df.High.iloc[k-2] + gap:

                last_bull_fvg_top = df.Low.iloc[k]
                last_bull_bar = k


            if df.High.iloc[k] < df.Low.iloc[k-2] - gap:

                last_bear_fvg_bot = df.High.iloc[k]
                last_bear_bar = k


        # ================= RULE-4 CORRECT ALIGNMENT =================

        smc_buy4 = False
        smc_sell4 = False


        if last_bear_bar is not None:

            if len(df) - last_bear_bar <= BOX_AGE_LIMIT:

                if last_bear_bar + 1 < len(df):

                    prev_exit = df.High.iloc[last_bear_bar + 1] < last_bear_fvg_bot

                    prev_bearish = df.Close.iloc[-2] < df.Open.iloc[-2]

                    curr_bullish = df.Close.iloc[-1] > df.Open.iloc[-1]

                    smc_buy4 = prev_exit and prev_bearish and curr_bullish


        if last_bull_bar is not None:

            if len(df) - last_bull_bar <= BOX_AGE_LIMIT:

                if last_bull_bar + 1 < len(df):

                    prev_exit = df.Low.iloc[last_bull_bar + 1] > last_bull_fvg_top

                    prev_bullish = df.Close.iloc[-2] > df.Open.iloc[-2]

                    curr_bearish = df.Close.iloc[-1] < df.Open.iloc[-1]

                    smc_sell4 = prev_exit and prev_bullish and curr_bearish


        # ================= FINAL SIGNAL =================

        buy_signal = smc_buy4 or (ha_buy and not ha_buy_prev)

        sell_signal = smc_sell4 or (ha_sell and not ha_sell_prev)


        if buy_signal:

            ce.append(stock)

        elif sell_signal:

            pe.append(stock)

        else:

            neutral.append(stock)


        progress.progress((i + 1) / total)


    st.success("Scan complete")


    st.subheader("🟢 CE Candidates")
    st.write(ce)


    st.subheader("🔴 PE Candidates")
    st.write(pe)


    st.subheader("⚪ Neutral")
    st.write(neutral)
