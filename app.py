import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

st.set_page_config(layout="wide")
st.title("F&O Scanner — Scored Signal Engine (CE/PE + BOS Boost)")

LEN1 = 5
LEN2 = 3
ATR_LEN = 3
ATR_THRESHOLD = 3.5

DEBUG = st.checkbox("Show CE/PE Trace Example")


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


# ================= STRUCTURE (LIGHT BOS / CHOCH) =================

def structure_score(df):
    recent_high = df.High.rolling(10).max().iloc[-2]
    recent_low = df.Low.rolling(10).min().iloc[-2]
    last_close = df.Close.iloc[-1]

    bos_up = last_close > recent_high
    choch_down = last_close < recent_low

    return bos_up, choch_down


# ================= ENGINE =================

def process(symbol, df):

    df = df.dropna().copy()
    df["ATR"] = atr(df)

    ha_open, ha_close = heikin_ashi(df)

    o1 = ha_open.ewm(span=LEN1, adjust=False).mean()
    c1 = ha_close.ewm(span=LEN1, adjust=False).mean()

    o2 = o1.ewm(span=LEN2, adjust=False).mean()
    c2 = c1.ewm(span=LEN2, adjust=False).mean()

    Hadiff = o2 - c2

    bos_up, choch_down = structure_score(df)

    ce_score = 0
    pe_score = 0

    ce_trace = None
    pe_trace = None

    for i in range(5, len(df)):

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
            ce_score = 1.0 + (0.5 if bos_up else 0)

            ce_trace = {
                "symbol": symbol,
                "type": "CE",
                "hadiff_prev": Hadiff.iloc[i-1],
                "hadiff_curr": Hadiff.iloc[i],
                "atr%": atr_pct,
                "bos_up": bos_up,
                "score": ce_score
            }


        if bearish:
            pe_score = 1.0 + (0.5 if choch_down else 0)

            pe_trace = {
                "symbol": symbol,
                "type": "PE",
                "hadiff_prev": Hadiff.iloc[i-1],
                "hadiff_curr": Hadiff.iloc[i],
                "atr%": atr_pct,
                "choch_down": choch_down,
                "score": pe_score
            }


    return ce_score, pe_score, ce_trace, pe_trace


# ================= RUN =================

if st.button("RUN SCAN"):

    ce_list = []
    pe_list = []

    ce_trace_final = None
    pe_trace_final = None

    for s in symbols:

        ticker = s + ".NS"
        if ticker not in data:
            continue

        df = data[ticker]

        ce_score, pe_score, ce_trace, pe_trace = process(s, df)


        if ce_score > 0:
            ce_list.append((s, ce_score))
            if ce_trace_final is None:
                ce_trace_final = ce_trace

        if pe_score > 0:
            pe_list.append((s, pe_score))
            if pe_trace_final is None:
                pe_trace_final = pe_trace


    # ================= SORT =================

    ce_list.sort(key=lambda x: x[1], reverse=True)
    pe_list.sort(key=lambda x: x[1], reverse=True)


    # ================= OUTPUT =================

    st.subheader("🟢 CE Candidates (Ranked)")
    st.write(ce_list)

    st.subheader("🔴 PE Candidates (Ranked)")
    st.write(pe_list)


    # ================= TRACE =================

    if DEBUG:

        st.divider()
        st.subheader("🧠 CE Example Trace")
        st.write(ce_trace_final)

        st.subheader("🧠 PE Example Trace")
        st.write(pe_trace_final)
