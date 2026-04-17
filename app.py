import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

st.set_page_config(layout="wide")
st.title("📊 NSE F&O SMC + Smoothed HA Scanner")

# ======================
# STRATEGY PARAMETERS
# ======================

SWING_LEN = 5
ATR_LEN = 3
ATR_THRESHOLD = 3.5
GAP_MULT = 0.5
BOX_AGE_LIMIT = 15

LEN1 = 5
LEN2 = 3


# ======================
# FETCH NSE F&O SYMBOLS
# ======================

@st.cache_data(ttl=86400)
def fetch_symbols():

    session = requests.Session()

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/"
    }

    session.get("https://www.nseindia.com", headers=headers)

    url = "https://www.nseindia.com/api/market-data-pre-open?key=FO"

    response = session.get(url, headers=headers)

    data = response.json()

    symbols = []

    for item in data["data"]:
        symbols.append(item["metadata"]["symbol"])

    return sorted(list(set(symbols)))


with st.spinner("Fetching F&O symbols..."):
    symbols = fetch_symbols()

st.success(f"{len(symbols)} symbols loaded")


# ======================
# DOWNLOAD DATA
# ======================

@st.cache_data
def load_data(symbols):

    tickers = [s + ".NS" for s in symbols]

    return yf.download(
        tickers=tickers,
        period="3mo",
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False
    )


with st.spinner("Downloading OHLC data..."):
    data = load_data(symbols)

st.success("OHLC ready")


# ======================
# HELPERS
# ======================

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def atr(df):

    tr = pd.concat([
        df.High - df.Low,
        abs(df.High - df.Close.shift()),
        abs(df.Low - df.Close.shift())
    ], axis=1).max(axis=1)

    return tr.rolling(ATR_LEN).mean()


def atr_percent(df):

    return (atr(df) / df.Close) * 100


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


# ======================
# SCANNER BUTTON
# ======================

if st.button("🚀 Run Scanner"):

    progress = st.progress(0)
    status = st.empty()

    ce = []
    pe = []
    neutral = []

    total = len(symbols)

    for i, stock in enumerate(symbols):

        status.text(f"Scanning {stock} ({i+1}/{total})")

        ticker = stock + ".NS"

        if ticker not in data:
            continue

        df = data[ticker].dropna()

        if len(df) < 50:
            continue


        # ======================
        # ATR FILTER
        # ======================

        df["ATR"] = atr(df)
        df["ATR%"] = (df["ATR"] / df.Close) * 100

        if df["ATR%"].iloc[-1] < ATR_THRESHOLD:

            neutral.append(stock)
            continue


        # ======================
        # SMOOTHED HA
        # ======================

        sOpen = ema(df.Open, LEN1)
        sClose = ema(df.Close, LEN1)
        sHigh = ema(df.High, LEN1)
        sLow = ema(df.Low, LEN1)

        ha_close = (sOpen + sHigh + sLow + sClose) / 4

        ha_open = ha_close.copy()

        ha_open.iloc[0] = (sOpen.iloc[0] + sClose.iloc[0]) / 2

        for j in range(1, len(df)):
            ha_open.iloc[j] = (
                ha_open.iloc[j-1] +
                ha_close.iloc[j-1]
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


        # ======================
        # STRUCTURE
        # ======================

        ph = pivot_high(df.High)
        pl = pivot_low(df.Low)

        lastH = ph.dropna().iloc[-1] if ph.dropna().size else None
        lastL = pl.dropna().iloc[-1] if pl.dropna().size else None


        bull_break = False
        bear_break = False


        if lastH is not None:
            bull_break = df.Close.iloc[-1] > lastH


        if lastL is not None:
            bear_break = df.Close.iloc[-1] < lastL


        # ======================
        # FVG DETECTION
        # ======================

        last_bull_fvg_top = None
        last_bull_fvg_bar = None

        last_bear_fvg_bot = None
        last_bear_fvg_bar = None


        for k in range(2, len(df)):

            gap = df["ATR"].iloc[k] * GAP_MULT


            if df.Low.iloc[k] > df.High.iloc[k-2] + gap:

                last_bull_fvg_top = df.Low.iloc[k]
                last_bull_fvg_bar = k


            if df.High.iloc[k] < df.Low.iloc[k-2] - gap:

                last_bear_fvg_bot = df.High.iloc[k]
                last_bear_fvg_bar = k


        bull_fvg_fresh = (

            last_bull_fvg_bar is not None
            and len(df) - last_bull_fvg_bar <= BOX_AGE_LIMIT

        )


        bear_fvg_fresh = (

            last_bear_fvg_bar is not None
            and len(df) - last_bear_fvg_bar <= BOX_AGE_LIMIT

        )


        broke_above_blue = (

            bull_fvg_fresh
            and df.Low.iloc[-2] > last_bull_fvg_top

        )


        broke_below_orange = (

            bear_fvg_fresh
            and df.High.iloc[-2] < last_bear_fvg_bot

        )


        prev_bullish = df.Close.iloc[-2] > df.Open.iloc[-2]
        prev_bearish = df.Close.iloc[-2] < df.Open.iloc[-2]


        smc_buy4 = (

            broke_below_orange
            and df.Close.iloc[-1] > df.Open.iloc[-1]
            and prev_bearish

        )


        smc_sell4 = (

            broke_above_blue
            and df.Close.iloc[-1] < df.Open.iloc[-1]
            and prev_bullish

        )


        # ======================
        # FINAL SIGNAL
        # ======================

        buy_signal = smc_buy4 or (ha_buy and not ha_buy_prev)

        sell_signal = smc_sell4 or (ha_sell and not ha_sell_prev)


        if buy_signal:

            ce.append(stock)

        elif sell_signal:

            pe.append(stock)

        else:

            neutral.append(stock)


        progress.progress((i + 1) / total)


    status.text("Scan complete")


    st.subheader("🟢 CE Candidates")
    st.write(ce)


    st.subheader("🔴 PE Candidates")
    st.write(pe)


    st.subheader("⚪ Neutral")
    st.write(neutral)
