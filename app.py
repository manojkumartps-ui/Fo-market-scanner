import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

st.set_page_config(layout="wide")
st.title("NSE F&O Strategy Engine (True PineScript Emulator)")


# ================= INPUTS =================

LEN1 = 5
LEN2 = 3
ATR_LEN = 3
ATR_THRESHOLD = 3.5

DEBUG = st.checkbox("Debug Mode")


# ================= F&O SYMBOLS =================

@st.cache_data(ttl=86400)
def get_fno():

    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    session.get("https://www.nseindia.com", headers=headers)

    url = "https://www.nseindia.com/api/market-data-pre-open?key=FO"
    data = session.get(url, headers=headers).json()

    return sorted(list(set(
        x["metadata"]["symbol"]
        for x in data["data"]
    )))


symbols = get_fno()
st.success(f"F&O symbols loaded: {len(symbols)}")


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


# ================= ATR =================

def calc_atr(df):

    tr = np.maximum(
        df.High - df.Low,
        np.maximum(
            abs(df.High - df.Close.shift()),
            abs(df.Low - df.Close.shift())
        )
    )

    return tr.rolling(ATR_LEN).mean()


# ================= TRUE PINE ENGINE =================

def run_engine(df, symbol):

    df = df.dropna().copy()

    if len(df) < 60:
        return None, None


    atr = calc_atr(df).values


    # -------- HA arrays --------
    ha_open = np.zeros(len(df))
    ha_close = np.zeros(len(df))

    ha_open[0] = (df.Open.iloc[0] + df.Close.iloc[0]) / 2
    ha_close[0] = (df.Open.iloc[0] + df.High.iloc[0] +
                   df.Low.iloc[0] + df.Close.iloc[0]) / 4


    # -------- Smoothed HA storage --------
    o2 = np.zeros(len(df))
    c2 = np.zeros(len(df))


    ce = False
    pe = False


    # ================= BAR BY BAR =================

    for i in range(1, len(df)):

        # ---- HA CALC ----
        ha_close[i] = (
            df.Open.iloc[i] +
            df.High.iloc[i] +
            df.Low.iloc[i] +
            df.Close.iloc[i]
        ) / 4

        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2


        # ---- SMOOTH HA (EMA style approximation) ----
        alpha1 = 2 / (LEN1 + 1)
        alpha2 = 2 / (LEN2 + 1)

        if i == 1:
            o2[i] = ha_open[i]
            c2[i] = ha_close[i]
        else:
            o2[i] = alpha2 * (alpha1 * ha_open[i] + (1-alpha1)*o2[i-1]) + (1-alpha2)*o2[i-1]
            c2[i] = alpha2 * (alpha1 * ha_close[i] + (1-alpha1)*c2[i-1]) + (1-alpha2)*c2[i-1]


        hadiff = o2[i] - c2[i]


        # ================= ATR FILTER =================

        atr_pct = atr[i] / df.Close.iloc[i] * 100 if atr[i] else 0

        if atr_pct < ATR_THRESHOLD:
            continue


        # ================= SIGNAL LOGIC =================

        # BUY
        if (
            df.Close.iloc[i] > df.Open.iloc[i]
            and hadiff < 0
            and o2[i-1] - c2[i-1] > 0
            and df.Close.iloc[i] > c2[i]
        ):
            ce = True


        # SELL
        if (
            df.Close.iloc[i] < df.Open.iloc[i]
            and hadiff > 0
            and o2[i-1] - c2[i-1] < 0
            and df.Close.iloc[i] < c2[i]
        ):
            pe = True


    return ce, pe


# ================= RUN =================

if st.button("RUN SCAN"):

    ce_list = []
    pe_list = []
    neutral_list = []

    progress = st.progress(0)

    for i, s in enumerate(symbols):

        ticker = s + ".NS"

        if ticker not in data:
            continue

        df = data[ticker]


        ce, pe = run_engine(df, s)


        if ce:
            ce_list.append(s)
        elif pe:
            pe_list.append(s)
        else:
            neutral_list.append(s)


        if DEBUG:
            st.write(s, "CE:", ce, "PE:", pe)


        progress.progress((i+1)/len(symbols))


    st.success("DONE")


    st.subheader("🟢 CE Candidates")
    st.write(ce_list)

    st.subheader("🔴 PE Candidates")
    st.write(pe_list)

    st.subheader("⚪ Neutral")
    st.write(neutral_list)
