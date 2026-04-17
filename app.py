import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time

st.set_page_config(page_title="SMC + Smoothed HA Scanner", layout="wide")

st.title("🏹 NSE F&O SMC + Smoothed HA Scanner")


# ================= PARAMETERS =================

SWING_LEN = 5
ATR_LEN = 3
ATR_FILTER = 2
GAP_THRESHOLD = 0.5
BOX_AGE_LIMIT = 15
IMPULSE_MULTIPLIER = 1.2

HA_LEN1 = 5
HA_LEN2 = 3


# ================= NSE F&O LIST =================

@st.cache_data(ttl=86400)
def get_fno_symbols():

    session = requests.Session()

    headers = {"User-Agent": "Mozilla/5.0"}

    session.get("https://www.nseindia.com", headers=headers)

    url = (
        "https://www.nseindia.com/api/"
        "equity-stockIndices?"
        "index=SECURITIES%20IN%20F%26O"
    )

    data = session.get(url, headers=headers).json()["data"]

    return [x["symbol"] for x in data]


symbols = get_fno_symbols()

st.sidebar.success(f"{len(symbols)} F&O symbols loaded")


# ================= ATR =================

def ATR(df, length):

    hl = df.High - df.Low
    hc = abs(df.High - df.Close.shift())
    lc = abs(df.Low - df.Close.shift())

    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    return tr.rolling(length).mean()


# ================= STRUCTURE =================

def detect_structure(df):

    ph = df.High.rolling(SWING_LEN * 2 + 1, center=True).max()
    pl = df.Low.rolling(SWING_LEN * 2 + 1, center=True).min()

    lastH = ph.iloc[-SWING_LEN]
    lastL = pl.iloc[-SWING_LEN]

    bull_break = df.Close.iloc[-1] > lastH
    bear_break = df.Close.iloc[-1] < lastL

    return bull_break, bear_break, lastH, lastL


# ================= SMOOTHED HA =================

def smoothed_ha(df):

    sOpen = df.Open.ewm(span=HA_LEN1).mean()
    sClose = df.Close.ewm(span=HA_LEN1).mean()
    sHigh = df.High.ewm(span=HA_LEN1).mean()
    sLow = df.Low.ewm(span=HA_LEN1).mean()

    ha_close = (sOpen + sHigh + sLow + sClose) / 4

    ha_open = [(sOpen.iloc[0] + sClose.iloc[0]) / 2]

    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha_close.iloc[i-1]) / 2)

    ha_open = pd.Series(ha_open, index=df.index)

    o2 = ha_open.ewm(span=HA_LEN2).mean()
    c2 = ha_close.ewm(span=HA_LEN2).mean()

    return o2, c2


# ================= FVG =================

def detect_fvg(df, atr):

    impulse = (df.High - df.Low) > atr * IMPULSE_MULTIPLIER

    bull_gap = (
        (df.Low > df.High.shift(2))
        &
        ((df.Low - df.High.shift(2)) > atr * GAP_THRESHOLD)
        &
        impulse.shift(1)
    )

    bear_gap = (
        (df.High < df.Low.shift(2))
        &
        ((df.Low.shift(2) - df.High) > atr * GAP_THRESHOLD)
        &
        impulse.shift(1)
    )

    return bull_gap, bear_gap


# ================= RULE 4 LOGIC =================

def fvg_retest(df, bull_gap, bear_gap):

    recent_bull = bull_gap.iloc[-BOX_AGE_LIMIT:]
    recent_bear = bear_gap.iloc[-BOX_AGE_LIMIT:]

    bull_retest = recent_bull.any() and df.Low.iloc[-1] <= df.Low.iloc[-3]

    bear_retest = recent_bear.any() and df.High.iloc[-1] >= df.High.iloc[-3]

    return bull_retest, bear_retest


# ================= SIGNAL ENGINE =================

def generate_signal(df):

    atr = ATR(df, ATR_LEN)

    atr_pct = atr.iloc[-1] / df.Close.iloc[-1] * 100

    st.write("ATR%:", round(atr_pct, 2))

    if atr_pct < ATR_FILTER:

        return "NO SIGNAL", "LOW VOLATILITY"


    o2, c2 = smoothed_ha(df)

    hadiff = o2 - c2


    ha_buy = (hadiff.iloc[-1] < 0) and (hadiff.iloc[-2] > 0)

    ha_sell = (hadiff.iloc[-1] > 0) and (hadiff.iloc[-2] < 0)


    bull_break, bear_break, lastH, lastL = detect_structure(df)

    st.write("Structure High:", lastH)
    st.write("Structure Low:", lastL)


    bull_gap, bear_gap = detect_fvg(df, atr)

    bull_retest, bear_retest = fvg_retest(df, bull_gap, bear_gap)


    smc_buy = bull_retest and bull_break

    smc_sell = bear_retest and bear_break


    if smc_buy and ha_buy:

        return "BUY", "SMC + HA CONFIRMED"


    if smc_sell and ha_sell:

        return "SELL", "SMC + HA CONFIRMED"


    return "NO SIGNAL", "NO ALIGNMENT"


# ================= DOWNLOAD =================

def download_batch(symbols):

    tickers = " ".join([s + ".NS" for s in symbols])

    return yf.download(
        tickers,
        period="3mo",
        interval="1d",
        group_by="ticker",
        progress=False
    )


# ================= UI =================

scan_depth = st.sidebar.slider(
    "Scan Depth",
    5,
    len(symbols),
    15
)


# ================= SCANNER =================

if st.button("🚀 Run Scan"):

    selected = symbols[:scan_depth]

    st.info("Downloading OHLC...")

    price_data = download_batch(selected)

    results = []

    progress = st.progress(0)


    for i, symbol in enumerate(selected):

        st.subheader(f"Analyzing {symbol}")

        try:

            df = price_data[symbol + ".NS"]

            signal, reason = generate_signal(df)

            price = round(df.Close.iloc[-1], 2)

            st.write("Signal:", signal)

            st.caption(reason)


            results.append({

                "Stock": symbol,
                "Price": price,
                "Signal": signal,
                "Reason": reason

            })


        except Exception as e:

            st.error(f"{symbol} failed → {e}")


        progress.progress((i + 1) / scan_depth)

        time.sleep(0.05)


    st.divider()

    st.subheader("📊 Consolidated Report")

    st.dataframe(pd.DataFrame(results), use_container_width=True)


st.caption("Engine: Corrected SMC + Smoothed HA + ATR Expansion Strategy")
