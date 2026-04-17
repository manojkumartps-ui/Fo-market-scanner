import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import time

st.set_page_config(page_title="NSE F&O Scanner", layout="wide")

st.title("🏹 NSE F&O Scanner (20-Day Heikin Ashi Engine)")


# ===============================
# FETCH NSE F&O SYMBOLS
# ===============================

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

st.sidebar.success(
    f"{len(symbols)} F&O symbols loaded"
)


# ===============================
# BATCH DOWNLOAD PRICE DATA
# ===============================

def batch_price_download(symbols):

    tickers = " ".join(
        [s + ".NS" for s in symbols]
    )

    df = yf.download(
        tickers,
        period="1mo",
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False
    )

    return df


# ===============================
# HEIKIN ASHI ENGINE
# ===============================

def compute_trend(df_symbol):

    if df_symbol.empty:

        return None, None

    ha_close = (
        df_symbol["Open"]
        + df_symbol["High"]
        + df_symbol["Low"]
        + df_symbol["Close"]
    ) / 4

    ha_open = [
        (
            df_symbol["Open"].iloc[0]
            + df_symbol["Close"].iloc[0]
        ) / 2
    ]

    for i in range(1, len(df_symbol)):

        ha_open.append(
            (
                ha_open[i - 1]
                + ha_close.iloc[i - 1]
            ) / 2
        )

    last_open = float(ha_open[-1])
    last_close = float(ha_close.iloc[-1])

    trend = (
        "Bullish 🟢"
        if last_close > last_open
        else "Bearish 🔴"
    )

    ltp = round(
        float(df_symbol["Close"].iloc[-1]),
        2
    )

    return trend, ltp


# ===============================
# UI CONTROL PANEL
# ===============================

scan_depth = st.sidebar.slider(
    "Scan Depth",
    5,
    len(symbols),
    20
)


# ===============================
# SCANNER ENGINE
# ===============================

if st.button("🚀 Run Scanner"):

    selected_symbols = symbols[:scan_depth]

    st.info("Downloading OHLC data batch...")

    price_data = batch_price_download(
        selected_symbols
    )

    results = []

    progress = st.progress(0)

    for i, symbol in enumerate(selected_symbols):

        try:

            df_symbol = price_data[symbol + ".NS"]

            trend, price = compute_trend(
                df_symbol
            )

            if trend is None:

                continue

            results.append(
                {
                    "Stock": symbol,
                    "Price": price,
                    "Trend": trend
                }
            )

            st.success(
                f"{symbol} → {trend} | ₹{price}"
            )

        except Exception as e:

            st.error(
                f"{symbol} failed → {e}"
            )

        progress.progress(
            (i + 1) / scan_depth
        )

        time.sleep(0.05)

    st.divider()

    st.subheader(
        "📊 Consolidated Market Report"
    )

    st.dataframe(
        pd.DataFrame(results),
        use_container_width=True
    )


st.caption(
    "Universe: Official NSE F&O list | Engine: 20-Day Heikin Ashi"
)
