import streamlit as st
import pandas as pd
import yfinance as yf
import requests

st.set_page_config(layout="wide")
st.title("NSE F&O Smoothed HA Scanner (Parity Fix Build)")


LEN1 = 5
LEN2 = 3
ATR_LEN = 3
ATR_THRESHOLD = 3.5
GAP_MULT = 0.5
BOX_AGE_LIMIT = 15


DEBUG = st.checkbox("Enable Debug Mode")


# ================= ATR =================

def atr(df):

    tr = pd.concat([
        df.High - df.Low,
        abs(df.High - df.Close.shift()),
        abs(df.Low - df.Close.shift())
    ], axis=1).max(axis=1)

    return tr.rolling(ATR_LEN).mean()


# ================= FETCH F&O SYMBOLS =================

@st.cache_data(ttl=86400)
def fetch_fno():

    session = requests.Session()

    headers = {"User-Agent": "Mozilla/5.0"}

    session.get("https://www.nseindia.com", headers=headers)

    url = "https://www.nseindia.com/api/market-data-pre-open?key=FO"

    data = session.get(url, headers=headers).json()

    return sorted(
        list(set(
            x["metadata"]["symbol"]
            for x in data["data"]
        ))
    )


symbols = fetch_fno()

st.success(f"{len(symbols)} F&O symbols loaded")


# ================= LOAD DATA =================

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

st.success("OHLC ready")


# ================= SCAN =================

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


        # ================= ATR FILTER =================

        df["ATR"] = atr(df)

        atr_pct = df["ATR"].iloc[-1] / df.Close.iloc[-1] * 100

        if atr_pct < ATR_THRESHOLD:

            neutral.append(symbol)
            continue


        # ================= SMOOTHED HA =================

        sOpen = df.Open.ewm(span=LEN1, adjust=False).mean()
        sClose = df.Close.ewm(span=LEN1, adjust=False).mean()
        sHigh = df.High.ewm(span=LEN1, adjust=False).mean()
        sLow = df.Low.ewm(span=LEN1, adjust=False).mean()


        ha_close = (sOpen + sHigh + sLow + sClose) / 4


        ha_open = ha_close.copy()

        ha_open.iloc[0] = (sOpen.iloc[0] + sClose.iloc[0]) / 2


        for j in range(1, len(df)):

            ha_open.iloc[j] = (
                ha_open.iloc[j - 1]
                + ha_close.iloc[j - 1]
            ) / 2


        o2 = ha_open.ewm(span=LEN2, adjust=False).mean()
        c2 = ha_close.ewm(span=LEN2, adjust=False).mean()


        Hadiff = o2 - c2


        # ================= TRUE CROSSOVER WINDOW =================

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


        # ================= RULE-4 PLACEHOLDER (same as earlier working version) =================

        if ha_buy:
            ce.append(symbol)

        elif ha_sell:
            pe.append(symbol)

        else:
            neutral.append(symbol)


        if DEBUG:

            st.write(symbol)
            st.write("ATR%", atr_pct)
            st.write("Hadiff[-1]", Hadiff.iloc[-1])
            st.write("Hadiff[-2]", Hadiff.iloc[-2])
            st.write("Hadiff[-3]", Hadiff.iloc[-3])
            st.write("Hadiff[-4]", Hadiff.iloc[-4])


        progress.progress((i + 1) / len(symbols))


    st.success("Scan complete")


    st.subheader("🟢 CE Candidates")
    st.write(ce)


    st.subheader("🔴 PE Candidates")
    st.write(pe)


    st.subheader("⚪ Neutral")
    st.write(neutral)
