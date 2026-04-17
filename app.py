import streamlit as st
import pandas as pd
import yfinance as yf
import requests

st.set_page_config(layout="wide")
st.title("NSE F&O Smoothed HA Scanner (True TradingView Version)")


LEN1 = 5
LEN2 = 3
ATR_LEN = 3
ATR_THRESHOLD = 3.5

DEBUG = st.checkbox("Enable Debug Mode")


# ================= ATR =================

def atr(df):

    tr = pd.concat([
        df.High - df.Low,
        abs(df.High - df.Close.shift()),
        abs(df.Low - df.Close.shift())
    ], axis=1).max(axis=1)

    return tr.rolling(ATR_LEN).mean()


# ================= FETCH F&O =================

@st.cache_data(ttl=86400)
def fetch_fno():

    session = requests.Session()

    headers = {"User-Agent": "Mozilla/5.0"}

    session.get("https://www.nseindia.com", headers=headers)

    url = "https://www.nseindia.com/api/market-data-pre-open?key=FO"

    data = session.get(url, headers=headers).json()

    return sorted(list(set(
        x["metadata"]["symbol"]
        for x in data["data"]
    )))


symbols = fetch_fno()

st.success(f"{len(symbols)} F&O symbols loaded")


# ================= DATA LOAD =================

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


# ================= BUILD TRUE HEIKIN ASHI =================

def build_heikin_ashi(df):

    ha = pd.DataFrame(index=df.index)

    ha["close"] = (
        df.Open +
        df.High +
        df.Low +
        df.Close
    ) / 4

    ha["open"] = ha["close"].copy()

    ha["open"].iloc[0] = (
        df.Open.iloc[0] +
        df.Close.iloc[0]
    ) / 2

    for i in range(1, len(df)):

        ha["open"].iloc[i] = (
            ha["open"].iloc[i-1] +
            ha["close"].iloc[i-1]
        ) / 2

    ha["high"] = ha[["open","close"]].join(df.High).max(axis=1)

    ha["low"] = ha[["open","close"]].join(df.Low).min(axis=1)

    return ha


# ================= RUN SCAN =================

if st.button("RUN SCAN"):

    ce = []
    pe = []
    neutral = []

    progress = st.progress(0)


    for i, symbol in enumerate(symbols):

        ticker = symbol + ".NS"

        if ticker not in data:
            continue

        df = data[ticker].dropna()

        if len(df) < 60:
            continue


        df["ATR"] = atr(df)

        atr_pct = df["ATR"].iloc[-1] / df.Close.iloc[-1] * 100

        if atr_pct < ATR_THRESHOLD:

            neutral.append(symbol)
            continue


        # ================= TRUE HA =================

        ha = build_heikin_ashi(df)


        # ================= SMOOTH HA =================

        ha_open_1 = ha["open"].ewm(span=LEN1, adjust=False).mean()

        ha_close_1 = ha["close"].ewm(span=LEN1, adjust=False).mean()


        o2 = ha_open_1.ewm(span=LEN2, adjust=False).mean()

        c2 = ha_close_1.ewm(span=LEN2, adjust=False).mean()


        Hadiff = o2 - c2


        ha_buy = (

            df.Close.iloc[-1] > df.Open.iloc[-1]

            and Hadiff.iloc[-1] < 0

            and (
                Hadiff.iloc[-2] > 0
                or Hadiff.iloc[-3] > 0
                or Hadiff.iloc[-4] > 0
            )

            and df.Close.iloc[-1] > c2.iloc[-1]

        )


        ha_sell = (

            df.Close.iloc[-1] < df.Open.iloc[-1]

            and Hadiff.iloc[-1] > 0

            and (
                Hadiff.iloc[-2] < 0
                or Hadiff.iloc[-3] < 0
                or Hadiff.iloc[-4] < 0
            )

            and df.Close.iloc[-1] < c2.iloc[-1]

        )


        if DEBUG:

            st.write(symbol)

            st.write("ATR%", atr_pct)

            st.write("Hadiff[-1]", Hadiff.iloc[-1])

            st.write("Hadiff[-2]", Hadiff.iloc[-2])

            st.write("Hadiff[-3]", Hadiff.iloc[-3])


        if ha_buy:

            ce.append(symbol)

        elif ha_sell:

            pe.append(symbol)

        else:

            neutral.append(symbol)


        progress.progress((i+1)/len(symbols))


    st.success("Scan complete")

    st.subheader("🟢 CE Candidates")

    st.write(ce)

    st.subheader("🔴 PE Candidates")

    st.write(pe)

    st.subheader("⚪ Neutral")

    st.write(neutral)
