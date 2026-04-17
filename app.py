import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from textblob import TextBlob
import time

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="NSE F&O Scanner",
    layout="wide"
)

st.title("🏹 NSE F&O Scanner (20-Day Heikin Ashi Engine)")

DEBUG_MODE = st.sidebar.checkbox(
    "Enable Debug Mode",
    True
)

# ======================================
# STAGE 1 — NSE F&O SYMBOL FETCH
# ======================================

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

    symbols = [
        item["symbol"]
        for item in json_data["data"]
    ]

    return symbols


symbols = get_fno_symbols()

if DEBUG_MODE:
    st.sidebar.success(
        f"{len(symbols)} F&O symbols fetched"
    )
    st.sidebar.write(
        "Sample:",
        symbols[:10]
    )

# ======================================
# STAGE 2 — PRICE DATA FETCH
# ======================================

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


# ======================================
# STAGE 3 — HEIKIN ASHI ENGINE
# ======================================

def compute_heikin_ashi(df):

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

    return ha_open, ha_close


# ======================================
# STAGE 4 — TREND DETECTOR
# ======================================

def detect_trend(ha_open, ha_close):

    last_open = float(ha_open[-1])
    last_close = float(ha_close.iloc[-1])

    if last_close > last_open:
        return "Bullish 🟢"

    return "Bearish 🔴"


# ======================================
# STAGE 5 — NEWS FETCH
# ======================================

def fetch_news(symbol):

    ticker = yf.Ticker(symbol + ".NS")

    news = ticker.news

    if news is None:
        return []

    titles = []

    for n in news:

        if isinstance(n, dict):

            title = n.get("title")

            if title:
                titles.append(title)

        if len(titles) == 5:
            break

    return titles


# ======================================
# STAGE 6 — SENTIMENT ENGINE
# ======================================

def sentiment_score(titles):

    if len(titles) == 0:
        return 0.0

    scores = []

    for t in titles:

        polarity = TextBlob(t).sentiment.polarity

        scores.append(polarity)

    return round(
        sum(scores) / len(scores),
        2
    )


# ======================================
# STAGE 7 — TRADE SIGNAL ENGINE
# ======================================

def trade_signal(trend, sentiment):

    if trend == "Bullish 🟢" and sentiment > 0.2:
        return "LONG ✅"

    if trend == "Bearish 🔴" and sentiment < -0.2:
        return "SHORT 🔻"

    return "AVOID ⚠️"


# ======================================
# CONTROL PANEL
# ======================================

scan_depth = st.sidebar.slider(
    "Scan Depth",
    5,
    150,
    10
)


# ======================================
# SCANNER ENGINE
# ======================================

if st.button("🚀 Run Scanner"):

    results = []

    for symbol in symbols[:scan_depth]:

        st.write(f"Analyzing {symbol}")

        try:

            df = fetch_price(symbol)

            if DEBUG_MODE:
                st.write(
                    "Candles:",
                    len(df)
                )

            ha_open, ha_close = compute_heikin_ashi(df)

            trend = detect_trend(
                ha_open,
                ha_close
            )

            titles = fetch_news(symbol)

            if DEBUG_MODE:
                st.write(
                    "Headlines:",
                    titles
                )

            sentiment = sentiment_score(
                titles
            )

            signal = trade_signal(
                trend,
                sentiment
            )

            ltp = round(
                float(df["Close"].iloc[-1]),
                2
            )

            st.success(
                f"{symbol} → {trend} | ₹{ltp} | Sentiment {sentiment}"
            )

            results.append(
                {
                    "Stock": symbol,
                    "Price": ltp,
                    "Trend": trend,
                    "Sentiment": sentiment,
                    "Signal": signal
                }
            )

            time.sleep(0.3)

        except Exception as e:

            st.error(
                f"{symbol} failed → {e}"
            )

    st.divider()

    st.subheader(
        "📊 Consolidated Market Report"
    )

    st.dataframe(
        pd.DataFrame(results),
        use_container_width=True
    )

st.caption(
    "Universe: Official NSE Securities in F&O | Engine: 20-Day Heikin Ashi + News Sentiment"
)
