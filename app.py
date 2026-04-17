import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from transformers import pipeline
import time

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(page_title="Pro NSE F&O Scanner", layout="wide")
st.title("🏹 NSE F&O Pro Scanner (20D Heikin Ashi)")

# -------------------------------
# STAGE 0: LOAD FINBERT AI
# -------------------------------

@st.cache_resource
def load_pro_ai():
    try:
        return pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )
    except:
        return None

finbert = load_pro_ai()

# -------------------------------
# STAGE 1: LEGIT NSE F&O LIST
# -------------------------------

@st.cache_data(ttl=86400)
def get_full_fno_list():

    """
    Fetch official NSE derivatives underlying securities list
    Source: NSE derivatives underlying tracker
    """

    try:

        url = "https://www.nseindia.com/api/underlying-information"

        headers = {
            "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        session = requests.Session()
        session.get(
            "https://www.nseindia.com",
            headers=headers
        )

        response = session.get(
            url,
            headers=headers
        )

        data = response.json()

        symbols = sorted(
            [
                x["symbol"]
                for x in data["data"]
                if x["symbol"]
            ]
        )

        return [s + ".NS" for s in symbols]

    except Exception as e:

        st.warning(
            f"Live NSE F&O fetch failed ({e}). Using fallback basket."
        )

        return [
            "RELIANCE.NS",
            "HDFCBANK.NS",
            "ICICIBANK.NS",
            "INFY.NS",
            "TCS.NS",
            "SBIN.NS",
            "AXISBANK.NS",
            "LT.NS",
            "MARUTI.NS",
            "BAJFINANCE.NS"
        ]

# -------------------------------
# STAGE 2: HEIKIN ASHI ENGINE
# -------------------------------

def calculate_ha_20d(symbol):

    try:

        df = yf.download(
            symbol,
            period="1mo",
            interval="1d",
            progress=False
        )

        if df.empty or len(df) < 10:
            return "NO_DATA", 0, "Price fetch failed"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

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

        current_open = ha_open[-1]
        current_close = ha_close.iloc[-1]

        ltp = float(df["Close"].iloc[-1])

        trend = (
            "Bullish 🟢"
            if current_close > current_open
            else "Bearish 🔴"
        )

        return trend, round(ltp, 2), "Success"

    except Exception as e:

        return "ERROR", 0, str(e)

# -------------------------------
# STAGE 3: FINBERT SENTIMENT
# -------------------------------

def get_ai_sentiment(symbol):

    if finbert is None:
        return 0.0, "AI Not Loaded"

    try:

        ticker = yf.Ticker(symbol)

        news = ticker.news

        if not news:
            return 0.0, "No News Found"

        titles = [
            n["title"]
            for n in news[:5]
            if "title" in n
        ]

        if len(titles) == 0:
            return 0.0, "No Valid Headlines"

        results = finbert(titles)

        mapping = {
            "positive": 1,
            "neutral": 0,
            "negative": -1
        }

        scores = [
            mapping[r["label"].lower()] * r["score"]
            for r in results
        ]

        avg_score = round(
            sum(scores) / len(scores),
            2
        )

        return avg_score, f"{len(titles)} headlines analyzed"

    except Exception as e:

        return 0.0, str(e)

# -------------------------------
# STAGE 4: TRADE SIGNAL ENGINE
# -------------------------------

def generate_trade_signal(trend, sentiment):

    if trend == "Bullish 🟢" and sentiment > 0.2:
        return "LONG ✅"

    if trend == "Bearish 🔴" and sentiment < -0.2:
        return "SHORT 🔻"

    return "AVOID ⚠️"

# -------------------------------
# UI CONTROL PANEL
# -------------------------------

scan_limit = st.sidebar.slider(
    "Scan Depth",
    5,
    100,
    15
)

# -------------------------------
# SCANNER ENGINE
# -------------------------------

if st.button("🚀 Start Market-Wide Scan"):

    symbols = get_full_fno_list()

    targets = symbols[:scan_limit]

    results = []

    for s in targets:

        with st.expander(
            f"Analyzing {s}",
            expanded=True
        ):

            col1, col2 = st.columns(2)

            with col1:

                trend, price, msg1 = calculate_ha_20d(s)

                st.success(
                    f"Trend: {trend} | ₹{price}"
                )

            with col2:

                sentiment, msg2 = get_ai_sentiment(s)

                st.info(
                    f"AI Score: {sentiment}"
                )

                st.caption(msg2)

            signal = generate_trade_signal(
                trend,
                sentiment
            )

            results.append(
                {
                    "Stock": s.replace(".NS", ""),
                    "Price": price,
                    "Trend": trend,
                    "AI Score": sentiment,
                    "Signal": signal
                }
            )

            time.sleep(0.4)

    st.divider()

    st.subheader(
        "📊 Consolidated Market Report"
    )

    st.dataframe(
        pd.DataFrame(results),
        use_container_width=True
    )

st.caption(
    "Scanner uses NSE derivatives universe + 20-day Heikin Ashi + FinBERT sentiment."
)
