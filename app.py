import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

st.set_page_config(layout="wide")
st.title("SMC + Smoothed HA DEBUG Scanner")

SWING_LEN = 5
ATR_LEN = 3
ATR_THRESHOLD = 3.5
GAP_MULT = 0.5
BOX_AGE_LIMIT = 15

LEN1 = 5
LEN2 = 3


# =====================
# FETCH F&O SYMBOLS
# =====================

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


# =====================
# DOWNLOAD DATA
# =====================

@st.cache_data
def load_data(symbols):

    tickers = [s + ".NS" for s in symbols]

    return yf.download(
        tickers=tickers,
        period="6mo",
        interval="1d",
        group_by="ticker",
        threads=True
    )


data = load_data(symbols)


# =====================
# HELPERS
# =====================

def ema(series, length):
    return series.ewm(span=length).mean()


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


# =====================
# RUN SCANNER
# =====================

if st.button("RUN DEBUG SCAN"):

    ce = []
    pe = []

    for stock in symbols[:40]:

        st.write("-------------")
        st.write(f"Scanning {stock}")

        ticker = stock + ".NS"

        if ticker not in data:
            continue

        df = data[ticker].dropna()

        if len(df) < 50:
            continue


        df["ATR"] = atr(df)
        df["ATR%"] = df["ATR"] / df.Close * 100

        atr_val = df["ATR%"].iloc[-1]

        st.write("ATR%", round(atr_val,2))

        if atr_val < ATR_THRESHOLD:

            st.write("❌ LOW VOLATILITY")
            continue


        # =====================
        # HA
        # =====================

        sOpen = ema(df.Open, LEN1)
        sClose = ema(df.Close, LEN1)
        sHigh = ema(df.High, LEN1)
        sLow = ema(df.Low, LEN1)

        ha_close = (sOpen+sHigh+sLow+sClose)/4

        ha_open = ha_close.copy()

        ha_open.iloc[0]=(sOpen.iloc[0]+sClose.iloc[0])/2

        for i in range(1,len(df)):

            ha_open.iloc[i]=(ha_open.iloc[i-1]+ha_close.iloc[i-1])/2


        o2=ema(ha_open,LEN2)
        c2=ema(ha_close,LEN2)

        Hadiff=o2-c2

        st.write("Hadiff now",round(Hadiff.iloc[-1],4))
        st.write("Hadiff[-3]",round(Hadiff.iloc[-4],4))


        # =====================
        # PIVOT
        # =====================

        ph=pivot_high(df.High)
        pl=pivot_low(df.Low)

        lastH=ph.dropna().iloc[-1] if ph.dropna().size else None
        lastL=pl.dropna().iloc[-1] if pl.dropna().size else None

        st.write("Pivot High",lastH)
        st.write("Pivot Low",lastL)


        # =====================
        # FVG CHECK
        # =====================

        bull_fvg=False
        bear_fvg=False

        for k in range(2,len(df)):

            gap=df["ATR"].iloc[k]*GAP_MULT

            if df.Low.iloc[k]>df.High.iloc[k-2]+gap:

                bull_fvg=True

            if df.High.iloc[k]<df.Low.iloc[k-2]-gap:

                bear_fvg=True


        st.write("Bull FVG exists",bull_fvg)
        st.write("Bear FVG exists",bear_fvg)


        # =====================
        # FINAL SIGNAL
        # =====================

        ha_buy=(Hadiff.iloc[-1]<0 and Hadiff.iloc[-4]>0)

        ha_sell=(Hadiff.iloc[-1]>0 and Hadiff.iloc[-4]<0)

        st.write("HA BUY",ha_buy)
        st.write("HA SELL",ha_sell)


        if ha_buy:

            ce.append(stock)

        elif ha_sell:

            pe.append(stock)


    st.write("CE",ce)
    st.write("PE",pe)
