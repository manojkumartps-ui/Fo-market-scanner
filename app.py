import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

st.set_page_config(layout="wide")
st.title("F&O Scanner — Last Valid Signal Engine (CE/PE)")

LEN1 = 5
LEN2 = 3
ATR_LEN = 3
ATR_THRESHOLD = 3.5


# ================= F&O LIST =================

@st.cache_data(ttl=86400)
def get_fno():
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    session.get("https://www.nseindia.com", headers=headers)

    url = "https://www.nseindia.com/api/market-data-pre-open?key=FO"
    data = session.get(url, headers=headers).json()

    return sorted(set(x["metadata"]["symbol"] for x in data["data"]))


symbols = get_fno()


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

def atr(df):
    tr = pd.concat([
        df.High - df.Low,
        abs(df.High - df.Close.shift()),
        abs(df.Low - df.Close.shift())
    ], axis=1).max(axis=1)

    return tr.rolling(ATR_LEN).mean()


# ================= HEIKIN ASHI =================

def heikin_ashi(df):
    ha_close = (df.Open + df.High + df.Low + df.Close) / 4

    ha_open = np.zeros(len(df))
    ha_open[0] = (df.Open.iloc[0] + df.Close.iloc[0]) / 2

    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2

    return pd.Series(ha_open, index=df.index), pd.Series(ha_close, index=df.index)


# ================= SMOOTHED HA =================

def smoothed_ha(df):
    ha_open, ha_close = heikin_ashi(df)

    o1 = ha_open.ewm(span=LEN1, adjust=False).mean()
    c1 = ha_close.ewm(span=LEN1, adjust=False).mean()

    o2 = o1.ewm(span=LEN2, adjust=False).mean()
    c2 = c1.ewm(span=LEN2, adjust=False).mean()

    return o2, c2


# ================= LAST VALID SIGNAL ENGINE =================

def evaluate_last_signal(df):

    df = df.dropna().copy()
    df["ATR"] = atr(df)

    o2, c2 = smoothed_ha(df)
    Hadiff = o2 - c2

    last_signal = "NONE"
    last_trace = None

    # scan full history (THIS is key fix)
    for i in range(2, len(df)):

        atr_pct = df.ATR.iloc[i] / df.Close.iloc[i] * 100
        if atr_pct < ATR_THRESHOLD:
            continue


        bullish = (
            Hadiff.iloc[i-1] <= 0 and
            Hadiff.iloc[i] > 0 and
            df.Close.iloc[i] > df.Open.iloc[i]
        )

        bearish = (
            Hadiff.iloc[i-1] >= 0 and
            Hadiff.iloc[i] < 0 and
            df.Close.iloc[i] < df.Open.iloc[i]
        )


        if bullish:
            last_signal = "CE"
            last_trace = {
                "bar_index": i,
                "hadiff_prev": float(Hadiff.iloc[i-1]),
                "hadiff_curr": float(Hadiff.iloc[i]),
                "close": float(df.Close.iloc[i]),
                "open": float(df.Open.iloc[i]),
                "atr%": float(atr_pct)
            }


        if bearish:
            last_signal = "PE"
            last_trace = {
                "bar_index": i,
                "hadiff_prev": float(Hadiff.iloc[i-1]),
                "hadiff_curr": float(Hadiff.iloc[i]),
                "close": float(df.Close.iloc[i]),
                "open": float(df.Open.iloc[i]),
                "atr%": float(atr_pct)
            }


    return last_signal, last_trace


# ================= RUN =================

if st.button("RUN SCAN"):

    ce_list = []
    pe_list = []
    neutral = []

    ce_example = None
    pe_example = None

    for s in symbols:

        ticker = s + ".NS"
        if ticker not in data:
            continue

        df = data[ticker]

        signal, trace = evaluate_last_signal(df)


        if signal == "CE":
            ce_list.append(s)
            if ce_example is None:
                ce_example = {s: trace}

        elif signal == "PE":
            pe_list.append(s)
            if pe_example is None:
                pe_example = {s: trace}

        else:
            neutral.append(s)


    # ================= OUTPUT =================

    st.subheader("🟢 CE Candidates (Last Valid Signal)")
    st.write(ce_list)

    st.subheader("🔴 PE Candidates (Last Valid Signal)")
    st.write(pe_list)

    st.subheader("⚪ Neutral")
    st.write(neutral)


    # ================= TRACE =================

    st.divider()

    st.subheader("🧠 CE Example Trace")
    st.write(ce_example)

    st.subheader("🧠 PE Example Trace")
    st.write(pe_example)
