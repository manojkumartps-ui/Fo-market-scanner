import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

st.set_page_config(layout="wide")
st.title("SMC + Smoothed HA Scanner (TradingView Parity Build)")


# ================= PARAMETERS =================

LEN1 = 5
LEN2 = 3

ATR_LEN = 3
ATR_THRESHOLD = 3.5

GAP_MULT = 0.5
BOX_AGE_LIMIT = 15

DEBUG_MODE = st.checkbox("Enable Debug Mode")


# ================= TRUE TRADINGVIEW EMA =================

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


# ================= FETCH NSE SYMBOLS =================

@st.cache_data(ttl=86400)
def fetch_symbols():

    session = requests.Session()

    headers = {"User-Agent": "Mozilla/5.0"}

    session.get("https://www.nseindia.com", headers=headers)

    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

    df = pd.read_csv(url)

    return df["SYMBOL"].tolist()


symbols = fetch_symbols()

st.success(f"{len(symbols)} NSE symbols loaded")


# ================= DOWNLOAD DATA =================

@st.cache_data
def load_data(symbols):

    tickers = [x + ".NS" for x in symbols]

    return yf.download(
        tickers=tickers,
        period="6mo",
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False
    )


data = load_data(symbols)

st.success("OHLC data ready")


# ================= RUN SCAN =================

if st.button("RUN SCAN"):

    ce = []
    pe = []
    neutral = []

    progress = st.progress(0)

    total = len(symbols)


    for i, symbol in enumerate(symbols):

        ticker = symbol + ".NS"

        if ticker not in data:
            continue

        df = data[ticker].dropna()

        if len(df) < 60:
            continue


        # ================= ATR FILTER =================

        df["ATR"] = atr(df)

        atr_percent = df["ATR"].iloc[-1] / df.Close.iloc[-1] * 100

        if atr_percent < ATR_THRESHOLD:

            neutral.append(symbol)

            continue


        # ================= SMOOTHED HA =================

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


        # ================= EXACT PINE PRECEDENCE BUY =================

        ha_buy = (

            (
                df.Close.iloc[-1] > df.Open.iloc[-1]
                and Hadiff.iloc[-1] < 0
                and Hadiff.iloc[-2] > 0
            )

            or

            Hadiff.iloc[-3] > 0

            or

            (
                Hadiff.iloc[-4] > 0
                and df.Close.iloc[-1] > c2.iloc[-1]
            )

        )


        # ================= EXACT PINE PRECEDENCE SELL =================

        ha_sell = (

            (
                df.Close.iloc[-1] < df.Open.iloc[-1]
                and Hadiff.iloc[-1] > 0
                and Hadiff.iloc[-2] < 0
            )

            or

            Hadiff.iloc[-3] < 0

            or

            (
                Hadiff.iloc[-4] < 0
                and df.Close.iloc[-1] < c2.iloc[-1]
            )

        )


        # ================= FVG DETECTION =================

        last_bull_fvg_top = None
        last_bull_bar = None

        last_bear_fvg_bot = None
        last_bear_bar = None


        for k in range(2, len(df)):

            gap = df["ATR"].iloc[k] * GAP_MULT


            if df.Low.iloc[k] > df.High.iloc[k - 2] + gap:

                last_bull_fvg_top = df.Low.iloc[k]

                last_bull_bar = k


            if df.High.iloc[k] < df.Low.iloc[k - 2] - gap:

                last_bear_fvg_bot = df.High.iloc[k]

                last_bear_bar = k


        smc_buy4 = False
        smc_sell4 = False


        # ================= RULE-4 ALIGNMENT =================

        if last_bear_bar is not None:

            if len(df) - last_bear_bar <= BOX_AGE_LIMIT:

                prev_exit = df.High.iloc[-2] < last_bear_fvg_bot

                prev_bearish = df.Close.iloc[-2] < df.Open.iloc[-2]

                curr_bullish = df.Close.iloc[-1] > df.Open.iloc[-1]

                smc_buy4 = prev_exit and prev_bearish and curr_bullish


        if last_bull_bar is not None:

            if len(df) - last_bull_bar <= BOX_AGE_LIMIT:

                prev_exit = df.Low.iloc[-2] > last_bull_fvg_top

                prev_bullish = df.Close.iloc[-2] > df.Open.iloc[-2]

                curr_bearish = df.Close.iloc[-1] < df.Open.iloc[-1]

                smc_sell4 = prev_exit and prev_bullish and curr_bearish


        # ================= FINAL SIGNAL =================

        buy_signal = smc_buy4 or ha_buy

        sell_signal = smc_sell4 or ha_sell


        if DEBUG_MODE:

            st.write(symbol)

            st.write("ATR%", round(atr_percent, 2))

            st.write("Hadiff[-1]", Hadiff.iloc[-1])

            st.write("Hadiff[-2]", Hadiff.iloc[-2])

            st.write("Hadiff[-3]", Hadiff.iloc[-3])

            st.write("Hadiff[-4]", Hadiff.iloc[-4])

            st.write("HA BUY", ha_buy)

            st.write("HA SELL", ha_sell)

            st.write("Rule4 BUY", smc_buy4)

            st.write("Rule4 SELL", smc_sell4)


        if buy_signal:

            ce.append(symbol)

        elif sell_signal:

            pe.append(symbol)

        else:

            neutral.append(symbol)


        progress.progress((i + 1) / total)


    st.success("Scan complete")


    st.subheader("🟢 CE Candidates")

    st.write(ce)


    st.subheader("🔴 PE Candidates")

    st.write(pe)


    st.subheader("⚪ Neutral")

    st.write(neutral)
