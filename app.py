import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO

# ================= PAGE CONFIG =================
st.set_page_config(layout="wide")
st.title("F&O Scanner — Momentum (SHA) + Structure (SMC)")

# ================= SETTINGS =================
LEN1 = 3
LEN2 = 2
SMC_LOOKBACK = 10
SCAN_LIMIT = 100

# ================= F&O FETCH =================
@st.cache_data(ttl=86400)
def get_fno():
    """
    Fetch F&O stock symbols from NSE official archive CSV.
    No fallback. Fails loudly if request fails.
    """
    url = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"

    try:
        with st.spinner("Fetching F&O symbols from NSE..."):
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            df = pd.read_csv(StringIO(response.text))

            # Clean columns
            df.columns = [col.strip() for col in df.columns]

            if "SYMBOL" not in df.columns:
                raise ValueError("SYMBOL column missing in NSE CSV")

            symbols = df["SYMBOL"].dropna().astype(str).unique().tolist()

            # Remove index contracts
            exclude = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}
            symbols = [s for s in symbols if s not in exclude]

            if len(symbols) == 0:
                raise ValueError("No F&O symbols found")

            return sorted(symbols)

    except Exception as e:
        st.error(f"F&O fetch failed: {e}")
        st.stop()


symbols = get_fno()

st.success(f"Loaded {len(symbols)} F&O stocks")

# ================= MARKET DATA LOAD =================
@st.cache_data(ttl=300)
def load_data(symbols):
    """
    Download OHLCV data from Yahoo Finance
    """
    tickers = [s + ".NS" for s in symbols[:SCAN_LIMIT]]

    with st.spinner("Downloading market data..."):
        data = yf.download(
            tickers=tickers,
            period="6mo",
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False
        )

    return data


data = load_data(symbols)

# ================= SIGNAL ENGINE =================
def evaluate_dual(df):
    df = df.dropna().copy()

    if len(df) < 20:
        return "NEUTRAL", None

    # HEIKIN ASHI
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4

    ha_open = np.zeros(len(df))
    ha_open[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2

    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2

    # SMOOTHED HA
    o2 = (
        pd.Series(ha_open)
        .ewm(span=LEN1, adjust=False)
        .mean()
        .ewm(span=LEN2, adjust=False)
        .mean()
    )

    c2 = (
        pd.Series(ha_close)
        .ewm(span=LEN1, adjust=False)
        .mean()
        .ewm(span=LEN2, adjust=False)
        .mean()
    )

    hadiff = o2 - c2
    i = len(df) - 1

    # SHA MOMENTUM
    sha_buy = (
        hadiff.iloc[i - 1] <= 0
        and hadiff.iloc[i] > 0
        and df["Close"].iloc[i] > df["Open"].iloc[i]
    )

    sha_sell = (
        hadiff.iloc[i - 1] >= 0
        and hadiff.iloc[i] < 0
        and df["Close"].iloc[i] < df["Open"].iloc[i]
    )

    # SMC BREAKOUT
    swing_high = df["High"].iloc[-SMC_LOOKBACK:-1].max()
    swing_low = df["Low"].iloc[-SMC_LOOKBACK:-1].min()

    smc_buy = (
        df["Close"].iloc[i] > swing_high
        and c2.iloc[i] > o2.iloc[i]
    )

    smc_sell = (
        df["Close"].iloc[i] < swing_low
        and c2.iloc[i] < o2.iloc[i]
    )

    # FINAL SIGNAL
    if sha_buy and smc_buy:
        return "BUY", {"Type": "STRONG"}
    elif sha_sell and smc_sell:
        return "SELL", {"Type": "STRONG"}
    elif sha_buy:
        return "BUY", {"Type": "Momentum"}
    elif smc_buy:
        return "BUY", {"Type": "SMC"}
    elif sha_sell:
        return "SELL", {"Type": "Momentum"}
    elif smc_sell:
        return "SELL", {"Type": "SMC"}

    return "NEUTRAL", None


# ================= RUN SCAN =================
if st.button("RUN SCAN"):
    buy_list = []
    sell_list = []

    with st.spinner("Running scanner..."):
        for s in symbols[:SCAN_LIMIT]:
            ticker = s + ".NS"

            try:
                if ticker not in data.columns.levels[0]:
                    continue

                df = data[ticker].dropna()

                signal, trace = evaluate_dual(df)

                if signal == "BUY":
                    buy_list.append({
                        "Symbol": s,
                        "Signal Type": trace["Type"]
                    })

                elif signal == "SELL":
                    sell_list.append({
                        "Symbol": s,
                        "Signal Type": trace["Type"]
                    })

            except Exception:
                continue

    # DISPLAY
    st.subheader(f"🟢 BUY Candidates ({len(buy_list)})")
    if buy_list:
        st.dataframe(pd.DataFrame(buy_list), use_container_width=True)
    else:
        st.write("No BUY signals")

    st.subheader(f"🔴 SELL Candidates ({len(sell_list)})")
    if sell_list:
        st.dataframe(pd.DataFrame(sell_list), use_container_width=True)
    else:
        st.write("No SELL signals")data = load(symbols)

# ================= SIGNAL ENGINE =================

def evaluate_dual(df):
    df = df.dropna().copy()
    
    if len(df) < 20:
        return "NEUTRAL", None

    # HEIKIN ASHI
    ha_close = (df.Open + df.High + df.Low + df.Close) / 4
    ha_open = np.zeros(len(df))
    ha_open[0] = (df.Open.iloc[0] + df.Close.iloc[0]) / 2

    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2

    # SMOOTHING
    o2 = pd.Series(ha_open).ewm(span=LEN1, adjust=False).mean().ewm(span=LEN2, adjust=False).mean()
    c2 = pd.Series(ha_close).ewm(span=LEN1, adjust=False).mean().ewm(span=LEN2, adjust=False).mean()
    
    Hadiff = o2 - c2
    i = len(df) - 1

    # SHA Momentum
    sha_buy = (Hadiff.iloc[i-1] <= 0 and Hadiff.iloc[i] > 0 and df.Close.iloc[i] > df.Open.iloc[i])
    sha_sell = (Hadiff.iloc[i-1] >= 0 and Hadiff.iloc[i] < 0 and df.Close.iloc[i] < df.Open.iloc[i])

    # SMC Structure (simple breakout)
    swing_high = df['High'].iloc[-SMC_LOOKBACK:-1].max()
    swing_low = df['Low'].iloc[-SMC_LOOKBACK:-1].min()
    
    smc_buy = (df.Close.iloc[i] > swing_high and c2.iloc[i] > o2.iloc[i])
    smc_sell = (df.Close.iloc[i] < swing_low and c2.iloc[i] < o2.iloc[i])

    # COMBINED LOGIC (improved)
    if sha_buy and smc_buy:
        return "BUY", {"Type": "STRONG"}
    elif sha_sell and smc_sell:
        return "SELL", {"Type": "STRONG"}
    elif sha_buy:
        return "BUY", {"Type": "Momentum"}
    elif smc_buy:
        return "BUY", {"Type": "SMC"}
    elif sha_sell:
        return "SELL", {"Type": "Momentum"}
    elif smc_sell:
        return "SELL", {"Type": "SMC"}
    
    return "NEUTRAL", None

# ================= RUN =================

if st.button("RUN SCAN"):
    if not symbols:
        st.error("F&O symbols not loaded.")
    elif data.empty:
        st.error("Market data not loaded.")
    else:
        buy_list = []
        sell_list = []

        for s in symbols[:100]:
            ticker = s + ".NS"

            try:
                if ticker not in data.columns.levels[0]:
                    continue

                df = data[ticker].dropna()
                sig, trace = evaluate_dual(df)

                if sig == "BUY":
                    buy_list.append(f"{s} ({trace['Type']})")
                elif sig == "SELL":
                    sell_list.append(f"{s} ({trace['Type']})")

            except:
                continue

        # DISPLAY
        st.subheader(f"🟢 BUY Candidates ({len(buy_list)})")
        st.write(buy_list)

        st.subheader(f"🔴 SELL Candidates ({len(sell_list)})")
        st.write(sell_list)
