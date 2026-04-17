import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(layout="wide")

st.title("🏹 NSE F&O Smoothed HA + SMC Scanner")


# =============================
# PARAMETERS (FROM YOUR STRATEGY)
# =============================

SWING_LEN = 5
ATR_LEN = 3
ATR_THRESHOLD = 3.5
LEN1 = 5
LEN2 = 3


# =============================
# FETCH LIVE NSE F&O LIST
# =============================

@st.cache_data(ttl=86400)
def fetch_fno_symbols():

    url = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"

    df = pd.read_csv(url)

    symbols = df["SYMBOL"].dropna().unique()

    return sorted(symbols)


symbols = fetch_fno_symbols()

st.write(f"Fetched {len(symbols)} F&O symbols")


# =============================
# DATA DOWNLOAD
# =============================

@st.cache_data
def download_data(symbols):

    tickers = [s + ".NS" for s in symbols]

    return yf.download(
        tickers=tickers,
        period="3mo",
        interval="1d",
        group_by="ticker",
        progress=False
    )


data = download_data(symbols)


# =============================
# EMA FUNCTION
# =============================

def ema(series, length):

    return series.ewm(span=length, adjust=False).mean()


# =============================
# ATR %
# =============================

def atr_percent(df):

    tr = pd.concat([
        df.High - df.Low,
        abs(df.High - df.Close.shift()),
        abs(df.Low - df.Close.shift())
    ], axis=1).max(axis=1)

    atr = tr.rolling(ATR_LEN).mean()

    return (atr / df.Close) * 100


# =============================
# PIVOT DETECTION (LIKE PINE)
# =============================

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
# SCANNER ENGINE
# =============================

ce_candidates = []
pe_candidates = []
neutral = []


for stock in symbols:

    if stock + ".NS" not in data:

        continue


    st.write(f"Analyzing {stock}")

    df = data[stock + ".NS"].dropna()

    if len(df) < 40:

        continue


    # =============================
    # ATR FILTER
    # =============================

    df["ATR%"] = atr_percent(df)

    atr_last = df["ATR%"].iloc[-1]

    st.write("ATR%:", round(atr_last,2))

    if atr_last < ATR_THRESHOLD:

        neutral.append(stock)

        continue


    # =============================
    # SMOOTHED HA
    # =============================

    sOpen = ema(df.Open, LEN1)
    sClose = ema(df.Close, LEN1)
    sHigh = ema(df.High, LEN1)
    sLow = ema(df.Low, LEN1)

    ha_close = (sOpen + sHigh + sLow + sClose)/4

    ha_open = ha_close.copy()

    ha_open.iloc[0] = (sOpen.iloc[0] + sClose.iloc[0])/2

    for i in range(1,len(df)):

        ha_open.iloc[i] = (
            ha_open.iloc[i-1]
            +
            ha_close.iloc[i-1]
        ) / 2


    o2 = ema(ha_open, LEN2)
    c2 = ema(ha_close, LEN2)

    Hadiff = o2 - c2


    ha_buy = (

        (df.Close.iloc[-1] > df.Open.iloc[-1])

        and

        (Hadiff.iloc[-1] < 0)

        and

        (Hadiff.iloc[-4] > 0)

        and

        (df.Close.iloc[-1] > c2.iloc[-1])

    )


    ha_sell = (

        (df.Close.iloc[-1] < df.Open.iloc[-1])

        and

        (Hadiff.iloc[-1] > 0)

        and

        (Hadiff.iloc[-4] < 0)

        and

        (df.Close.iloc[-1] < c2.iloc[-1])

    )


    # =============================
    # STRUCTURE BREAK
    # =============================

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

        st.success("BUY → CE candidate")


    elif sell_signal:

        pe_candidates.append(stock)

        st.error("SELL → PE candidate")


    else:

        neutral.append(stock)


# =============================
# FINAL OUTPUT
# =============================

st.subheader("🟢 CE Candidates")
st.write(ce_candidates)

st.subheader("🔴 PE Candidates")
st.write(pe_candidates)

st.subheader("⚪ Neutral")
st.write(neutral)
