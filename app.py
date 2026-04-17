import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from textblob import TextBlob
import gc
import time

st.set_page_config(page_title="NSE F&O Scanner", layout="wide")

st.title("🏹 NSE F&O Scanner (Low-Memory Engine)")

DEBUG_MODE = st.sidebar.checkbox("Enable Debug Mode", True)


# ==============================
# NSE F&O SYMBOL FETCH
# ==============================

@st.cache_data(ttl=86400)
def get_fno_symbols():

    session = requests.Session()

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    session.get(
        "https://www.nseindia.com",
        headers=headers
    )

    url = (
        "https://www.nseindia.com/api/"
        "equity-stockIndices?index=SECURITIES%20IN%20F%26O"
    )

    response = session.get(
        url,
        headers=headers
    )

    response.raise_for_status()

    json_data = response.json()

    return [
        item["symbol"]
        for item in json_data["data"]
    ]


symbols = get_fno_symbols()

if DEBUG_MODE:

    st.sidebar.success(
        f"Fetched {len(symbols)} F&O symbols"
    )

    st.sidebar.write(symbols[:10])


# ==============================
# PRICE FETCH
# ==============================

def fetch_price(symbol):

    df = yf.download(
        symbol + ".NS",
        period="1mo",
        interval="1d",
        progress=False
    )

    if df.empty:
        raise Exception("Empty dataframe")

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = df.columns.get_level_values(0)

    if len(df) < 10:
        raise Exception("Insufficient candles")

    return df


# ==============================
# HEIKIN ASHI ENGINE
# ==============================

def compute_ha(df):

    ha_close = (
        df["Open"]
        + df["High"]
        + df["Low"]
        + df["Close"]
    ) / 4

    ha_open = [
        (
            df["Open"].iloc[0]
            + df["Close"].iloc[0]
        ) / 2
    ]

    for i in range(1, len(df)):

        ha_open.append(
            (
                ha_open[i - 1]
                + ha_close.iloc[i - 1]
            ) / 2
        )

    return ha_open[-1], ha_close.iloc[-1]


# ==============================
# NEWS SENTIMENT
# ==============================

def sentiment(symbol):

    ticker = yf.Ticker(symbol + ".NS")

    news = ticker.news

    if not news:
        return 0.0

    titles = []

    for n in news:

        title = n.get("title")

        if title:
            titles.append(title)

        if len(titles) == 5:
            break

    if not titles:
        return 0.0

    scores = []

    for t in titles:

        scores.append(
            TextBlob(t).sentiment.polarity
        )

    return round(sum(scores) / len(scores), 2)


# ==============================
# TREND ENGINE
# ==============================

def trend_engine(open_val, close_val):

    if float(close_val) > float(open_val):

        return "Bullish 🟢"

    return "Bearish 🔴"


# ==============================
# SIGNAL ENGINE
# ==============================

def signal_engine(trend, sentiment):

    if trend == "Bullish 🟢" and sentiment > 0.2:

        return "LONG ✅"

    if trend == "Bearish 🔴" and sentiment < -0.2:

        return "SHORT 🔻"

    return "AVOID ⚠️"


# ==============================
# CONTROL PANEL
# ==============================

scan_depth = st.sidebar.slider(
    "Scan Depth",
    5,
    len(symbols),
    10
)


# ==============================
# SCANNER LOOP (STREAM SAFE)
# ==============================

if st.button("🚀 Run Scanner"):

    results = []

    progress = st.progress(0)

    for i, symbol in enumerate(symbols[:scan_depth]):

        st.write(f"Analyzing {symbol}")

        try:

            df = fetch_price(symbol)

            open_val, close_val = compute_ha(df)

            trend = trend_engine(
                open_val,
                close_val
            )

            sent = sentiment(symbol)

            signal = signal_engine(
                trend,
                sent
            )

            ltp = round(
                float(df["Close"].iloc[-1]),
                2
            )

            st.success(
                f"{symbol} → {trend} | ₹{ltp} | Sentiment {sent}"
            )

            results.append(
                {
                    "Stock": symbol,
                    "Price": ltp,
                    "Trend": trend,
                    "Sentiment": sent,
                    "Signal": signal
                }
            )

            del df
            gc.collect()

        except Exception as e:

            st.error(
                f"{symbol} failed → {e}"
            )

        progress.progress((i + 1) / scan_depth)

        time.sleep(0.15)

    st.divider()

    st.dataframe(
        pd.DataFrame(results),
        use_container_width=True
    )
