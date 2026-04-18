import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

st.set_page_config(layout="wide")
st.title("F&O Scanner — Momentum (SHA) + Structure (SMC) Engine")

# ================= SETTINGS =================
LEN1, LEN2 = 3, 2  # Your preferred fast smoothing
SMC_LOOKBACK = 10  # Days to look back for Swing High/Low

# ================= F&O LIST (Stable Source) =================
@st.cache_data(ttl=86400)
def get_fno():
    url = "https://kite.trade"
    try:
        df = pd.read_csv(url)
        return sorted(df[df['segment'] == 'NFO-FUT']['name'].unique().tolist())
    except:
        return ["MARICO", "SUPREMEIND", "RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"]

symbols = get_fno()

# ================= DATA LOAD =================
@st.cache_data
def load(symbols):
    tickers = [s + ".NS" for s in symbols]
    return yf.download(tickers=tickers, period="6mo", interval="1d", group_by="ticker", threads=True, progress=False)

data = load(symbols)

# ================= MATH FUNCTIONS =================
def heikin_ashi(df):
    ha_close = (df.Open + df.High + df.Low + df.Close) / 4
    ha_open = np.zeros(len(df))
    ha_open[0] = (df.Open.iloc[0] + df.Close.iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
    return pd.Series(ha_open, index=df.index), pd.Series(ha_close, index=df.index)

def smoothed_ha(df):
    ha_open, ha_close = heikin_ashi(df)
    o2 = ha_open.ewm(span=LEN1).mean().ewm(span=LEN2).mean()
    c2 = ha_close.ewm(span=LEN1).mean().ewm(span=LEN2).mean()
    return o2, c2

# ================= DUAL SIGNAL ENGINE =================
def evaluate_dual(df):
    df = df.dropna().copy()
    if len(df) < 20: return "NEUTRAL", None

    # 1. SHA MOMENTUM (Original Logic)
    o2, c2 = smoothed_ha(df)
    Hadiff = o2 - c2
    i = len(df) - 1
    
    sha_buy = (Hadiff.iloc[i-1] <= 0 and Hadiff.iloc[i] > 0 and df.Close.iloc[i] > df.Open.iloc[i])
    sha_sell = (Hadiff.iloc[i-1] >= 0 and Hadiff.iloc[i] < 0 and df.Close.iloc[i] < df.Open.iloc[i])

    # 2. SMC STRUCTURE (BOS/ChoCH)
    # Breaking the high/low of the last 10 days
    swing_high = df['High'].iloc[-SMC_LOOKBACK:-1].max()
    swing_low = df['Low'].iloc[-SMC_LOOKBACK:-1].min()
    
    smc_buy = (df.Close.iloc[i] > swing_high and c2.iloc[i] > o2.iloc[i])
    smc_sell = (df.Close.iloc[i] < swing_low and c2.iloc[i] < o2.iloc[i])

    # Result Aggregation
    if sha_buy or smc_buy:
        return "BUY", {"Type": "Momentum" if sha_buy else "SMC BOS", "Price": float(df.Close.iloc[i])}
    if sha_sell or smc_sell:
        return "SELL", {"Type": "Momentum" if sha_sell else "SMC BOS", "Price": float(df.Close.iloc[i])}
    
    return "NEUTRAL", None

# ================= RUN & OUTPUT =================
if st.button("RUN DUAL SCAN"):
    buy_list, sell_list = [], []
    for s in symbols:
        ticker = s + ".NS"
        if ticker not in data or data[ticker].empty: continue
        signal, trace = evaluate_dual(data[ticker])
        if signal == "BUY": buy_list.append(f"{s} ({trace['Type']})")
        elif signal == "SELL": sell_list.append(f"{s} ({trace['Type']})")

    st.subheader(f"🟢 BUY Signals ({len(buy_list)})")
    st.write(buy_list)
    st.subheader(f"🔴 SELL Signals ({len(sell_list)})")
    st.write(sell_list)
