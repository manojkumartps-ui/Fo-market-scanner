import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import time

st.set_page_config(
    page_title="NSE F&O Scanner",
    layout="wide"
)

st.title("🏹 NSE F&O Scanner (20-Day Heikin Ashi Regime Engine)")


# ===============================
# STAGE 1 — FETCH NSE F&O LIST
# ===============================

@st.cache_data(ttl=86400)
def get_fno_symbols():

    session = requests.Session()

    headers = {
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    session.get(
        "https://www.nseindia.com",
        headers=headers
    )

    url = (
        "https://www.nseindia.com/api/"
        "equity-stockIndices?"
        "index=SECURITIES%20IN%20F%26O"
    )

    response = session.get(
        url,
        headers=headers
    )

    response.raise_for_status()

    data = response.json()["data"]

    symbols = [
        item["symbol"]
        for item in data
    ]

    return symbols


symbols = get_fno_symbols()

st.sidebar.success(
    f"{len(symbols)} F&O symbols loaded"
)


# ===============================
# STAGE 2 — BATCH PRICE DOWNLOAD
# ===============================

def batch_download(symbols):

    ticker_string = " ".join(
        [s + ".NS" for s in symbols]
    )

    df = yf.download(
        ticker_string,
        period="1mo",
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False
    )

    return df


# ===============================
# STAGE 3 — HEIKIN ASHI ENGINE
# TRUE 20-DAY REGIME VERSION
# ===============================

def compute_ha_regime(df_symbol, symbol):

    if df_symbol.empty:

        st.error(f"{symbol} no OHLC data")

        return None


    candle_count = len(df_symbol)

    st.write(f"{symbol} candles fetched:", candle_count)


    if candle_count < 20:

        st.warning(f"{symbol} insufficient candles")

        return None


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


    ha_open = pd.Series(
        ha_open,
        index=df_symbol.index
    )


    last20_open = ha_open.iloc[-20:]
    last20_close = ha_close.iloc[-20:]


    bullish_count = (
        last20_close > last20_open
    ).sum()


    bearish_count = 20 - bullish_count


    body_strength = (
        last20_close - last20_open
    ).mean()


    last_open = float(last20_open.iloc[-1])
    last_close = float(last20_close.iloc[-1])


    if bullish_count >= 14:

        regime = "Bullish 🟢"

    elif bearish_count >= 14:

        regime = "Bearish 🔴"

    else:

        regime = "Sideways ⚪"


    ltp = round(
        float(df_symbol["Close"].iloc[-1]),
        2
    )


    st.write(
        f"{symbol} bullish candles:",
        bullish_count
    )

    st.write(
        f"{symbol} bearish candles:",
        bearish_count
    )

    st.write(
        f"{symbol} avg body strength:",
        round(body_strength, 3)
    )

    st.write(
        f"{symbol} last HA candle:",
        "Bullish"
        if last_close > last_open
        else "Bearish"
    )


    return {

        "Stock": symbol,

        "Price": ltp,

        "Trend": regime,

        "BullishCount": bullish_count,

        "BearishCount": bearish_count,

        "BodyStrength": round(body_strength, 3)

    }


# ===============================
# UI CONTROL PANEL
# ===============================

scan_depth = st.sidebar.slider(

    "Scan Depth",

    5,

    len(symbols),

    15
)


# ===============================
# SCANNER ENGINE
# ===============================

if st.button("🚀 Run Market Scan"):

    selected = symbols[:scan_depth]

    st.info("Downloading OHLC batch...")

    price_data = batch_download(selected)

    results = []

    progress = st.progress(0)


    for i, symbol in enumerate(selected):

        st.subheader(f"Analyzing {symbol}")

        try:

            df_symbol = price_data[
                symbol + ".NS"
            ]

            output = compute_ha_regime(

                df_symbol,

                symbol

            )

            if output:

                results.append(output)

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

    "Universe: Official NSE F&O list | Engine: True 20-Day Heikin Ashi Regime Logic"

)
