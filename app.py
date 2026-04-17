import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time

st.set_page_config(page_title="SMC + Smoothed HA Options Scanner", layout="wide")

st.title("🏹 NSE F&O Smoothed HA + SMC Options Scanner")


#########################################
# STRATEGY PARAMETERS (FROM YOUR SCRIPT)
#########################################

SWING_LEN = 5
ATR_LEN = 3
ATR_FILTER = 2
GAP_THRESHOLD = 0.5
BOX_AGE_LIMIT = 15

HA_LEN1 = 5
HA_LEN2 = 3


#########################################
# NSE F&O SYMBOL FETCHER
#########################################

@st.cache_data(ttl=86400)
def get_fno_symbols():

    session = requests.Session()

    headers = {"User-Agent": "Mozilla/5.0"}

    session.get("https://www.nseindia.com", headers=headers)

    url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"

    data = session.get(url, headers=headers).json()["data"]

    return [x["symbol"] for x in data]


symbols = get_fno_symbols()

st.sidebar.success(f"{len(symbols)} F&O symbols loaded")


#########################################
# ATR CALCULATION
#########################################

def ATR(df, length):

    hl = df.High - df.Low
    hc = abs(df.High - df.Close.shift())
    lc = abs(df.Low - df.Close.shift())

    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    return tr.rolling(length).mean()


#########################################
# STRUCTURE DETECTION (PINE EQUIVALENT)
#########################################

def detect_structure(df):

    highs = df.High.values
    lows = df.Low.values

    lastH = None
    lastL = None

    for i in range(SWING_LEN, len(df) - SWING_LEN):

        if highs[i] == max(highs[i-SWING_LEN:i+SWING_LEN+1]):
            lastH = highs[i]

        if lows[i] == min(lows[i-SWING_LEN:i+SWING_LEN+1]):
            lastL = lows[i]

    if lastH is None or lastL is None:
        return False, False, None, None

    bull_break = df.Close.iloc[-1] > lastH
    bear_break = df.Close.iloc[-1] < lastL

    return bull_break, bear_break, lastH, lastL


#########################################
# SMOOTHED HEIKIN ASHI
#########################################

def smoothed_ha(df):

    sOpen = df.Open.ewm(span=HA_LEN1).mean()
    sClose = df.Close.ewm(span=HA_LEN1).mean()
    sHigh = df.High.ewm(span=HA_LEN1).mean()
    sLow = df.Low.ewm(span=HA_LEN1).mean()

    ha_close = (sOpen + sHigh + sLow + sClose) / 4

    ha_open = [(sOpen.iloc[0] + sClose.iloc[0]) / 2]

    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha_close.iloc[i-1]) / 2)

    ha_open = pd.Series(ha_open, index=df.index)

    o2 = ha_open.ewm(span=HA_LEN2).mean()
    c2 = ha_close.ewm(span=HA_LEN2).mean()

    hadiff = o2 - c2

    return hadiff


#########################################
# FVG DETECTION
#########################################

def detect_fvg(df, atr):

    bull_gap = (
        (df.Low > df.High.shift(2)) &
        ((df.Low - df.High.shift(2)) > atr * GAP_THRESHOLD)
    )

    bear_gap = (
        (df.High < df.Low.shift(2)) &
        ((df.Low.shift(2) - df.High) > atr * GAP_THRESHOLD)
    )

    return bull_gap, bear_gap


#########################################
# RULE-4 FVG RETEST CHECK
#########################################

def detect_fvg_retest(df, bull_gap, bear_gap):

    recent_bull = bull_gap.iloc[-BOX_AGE_LIMIT:]
    recent_bear = bear_gap.iloc[-BOX_AGE_LIMIT:]

    bull_retest = recent_bull.any() and df.Low.iloc[-1] < df.Low.iloc[-2]
    bear_retest = recent_bear.any() and df.High.iloc[-1] > df.High.iloc[-2]

    return bull_retest, bear_retest


#########################################
# SIGNAL ENGINE (FINAL CE / PE DECISION)
#########################################

def generate_signal(symbol, df):

    st.subheader(f"Analyzing {symbol}")

    atr = ATR(df, ATR_LEN)

    atr_pct = atr.iloc[-1] / df.Close.iloc[-1] * 100

    st.write("ATR%:", round(atr_pct, 2))

    if atr_pct < ATR_FILTER:

        st.warning("LOW VOLATILITY FILTER ACTIVE")

        return "NO SIGNAL"


    ####################################
    # STRUCTURE CHECK
    ####################################

    bull_break, bear_break, lastH, lastL = detect_structure(df)

    st.write("Structure High:", lastH)
    st.write("Structure Low:", lastL)


    ####################################
    # SMOOTHED HA CROSSOVER
    ####################################

    hadiff = smoothed_ha(df)

    ha_buy = hadiff.iloc[-1] < 0 and hadiff.iloc[-2] > 0
    ha_sell = hadiff.iloc[-1] > 0 and hadiff.iloc[-2] < 0

    st.write("HA Buy:", ha_buy)
    st.write("HA Sell:", ha_sell)


    ####################################
    # FVG CHECK
    ####################################

    bull_gap, bear_gap = detect_fvg(df, atr)

    bull_retest, bear_retest = detect_fvg_retest(df, bull_gap, bear_gap)

    st.write("Bull FVG present:", bull_gap.iloc[-5:].any())
    st.write("Bear FVG present:", bear_gap.iloc[-5:].any())

    st.write("Bull FVG retest:", bull_retest)
    st.write("Bear FVG retest:", bear_retest)


    ####################################
    # FINAL SIGNAL DECISION
    ####################################

    smc_buy = bull_break and bull_retest
    smc_sell = bear_break and bear_retest


    if smc_buy and ha_buy:

        st.success("CE CANDIDATE CONFIRMED")

        return "CE"


    if smc_sell and ha_sell:

        st.error("PE CANDIDATE CONFIRMED")

        return "PE"


    return "NO SIGNAL"


#########################################
# DATA DOWNLOAD
#########################################

def download_batch(symbols):

    tickers = " ".join([s + ".NS" for s in symbols])

    return yf.download(
        tickers,
        period="3mo",
        interval="1d",
        group_by="ticker",
        progress=False
    )


#########################################
# UI CONTROL
#########################################

scan_depth = st.sidebar.slider("Stocks to scan", 5, len(symbols), 12)


#########################################
# RUN SCANNER
#########################################

if st.button("🚀 Run Scanner"):

    selected = symbols[:scan_depth]

    st.info("Downloading OHLC data...")

    price_data = download_batch(selected)

    ce_list = []
    pe_list = []
    neutral_list = []

    progress = st.progress(0)

    for i, symbol in enumerate(selected):

        try:

            df = price_data[symbol + ".NS"]

            signal = generate_signal(symbol, df)

            price = round(df.Close.iloc[-1], 2)

            if signal == "CE":
                ce_list.append([symbol, price])

            elif signal == "PE":
                pe_list.append([symbol, price])

            else:
                neutral_list.append([symbol, price])

        except Exception as e:

            st.error(f"{symbol} failed → {e}")

        progress.progress((i + 1) / scan_depth)

        time.sleep(0.05)


    #########################################
    # FINAL OUTPUT TABLES
    #########################################

    st.divider()

    st.subheader("🟢 CE Candidates")

    st.dataframe(pd.DataFrame(ce_list, columns=["Stock", "Price"]))

    st.subheader("🔴 PE Candidates")

    st.dataframe(pd.DataFrame(pe_list, columns=["Stock", "Price"]))

    st.subheader("⚪ Neutral")

    st.dataframe(pd.DataFrame(neutral_list, columns=["Stock", "Price"]))
