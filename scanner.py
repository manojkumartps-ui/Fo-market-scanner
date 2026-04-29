import pandas as pd
import numpy as np

FILE = "data/nse_fno_ohlc.csv"

# -----------------------------
# Heikin Ashi - YOUR ORIGINAL LOGIC
# -----------------------------
def heikin_ashi(df):
    ha = pd.DataFrame(index=df.index)
    ha["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = [(df["Open"].iloc[0] + df["Close"].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha["HA_Close"].iloc[i - 1]) / 2)
    ha["HA_Open"] = ha_open
    return ha

# -----------------------------
# EMA - YOUR ORIGINAL LOGIC
# -----------------------------
def add_ema(df):
    df["EMA_HA_Open"] = df["HA_Open"].ewm(span=5, adjust=False).mean()
    df["EMA_HA_Close"] = df["HA_Close"].ewm(span=5, adjust=False).mean()
    return df

# -----------------------------
# Strategy - YOUR ORIGINAL LOGIC
# -----------------------------
def check_stock(df):
    if len(df) < 30:
        return False

    latest = df.iloc[-1]
    d_3 = df.iloc[-4]

    weekly = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()

    if len(weekly) < 5:
        return False

    ha_w = heikin_ashi(weekly)
    weekly = pd.concat([weekly, ha_w], axis=1)
    weekly = add_ema(weekly)

    w_latest = weekly.iloc[-1]
    w_3 = weekly.iloc[-4]

    return (
        latest["Close"] > latest["Open"] and
        latest["Close"] > latest["EMA_HA_Close"] and
        latest["EMA_HA_Open"] < latest["EMA_HA_Close"] and

        d_3["EMA_HA_Open"] > d_3["EMA_HA_Close"] and

        w_latest["Close"] > w_latest["Open"] and
        w_latest["Close"] > w_latest["EMA_HA_Close"] and
        w_latest["EMA_HA_Open"] < w_latest["EMA_HA_Close"] and

        w_3["EMA_HA_Open"] > w_3["EMA_HA_Close"]
    )

# -----------------------------
# Main Scanner - UPDATED PARSING FOR YOUR CSV
# -----------------------------
def run_scanner():
    # 1. Skip the 'Price' row (Row 0), use Ticker (Row 1) and Metric (Row 2) as header
    # This aligns the CSV structure shown in your image
    raw = pd.read_csv(FILE, header=[1, 2], index_col=0, skiprows=[0])

    # 2. Fix the Date Warning & clean the index
    # We use format='mixed' or dayfirst to handle the Excel/CSV date strings
    raw.index = pd.to_datetime(raw.index, errors="coerce", dayfirst=True)
    raw = raw[~raw.index.isna()]
    raw = raw.sort_index()

    results = []
    # Tickers are in the first level of the multi-index columns
    tickers = raw.columns.levels[0]

    for stock in tickers:
        try:
            df = raw[stock].copy()
            df = df[["Open", "High", "Low", "Close"]]
            
            # Numeric conversion to ensure no string data breaks the logic
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df = df.dropna()
            df = df.sort_index()

            if len(df) < 30:
                continue

            ha = heikin_ashi(df)
            df = pd.concat([df, ha], axis=1)
            df = add_ema(df)

            if check_stock(df):
                results.append(stock)

        except Exception as e:
            # print(f"Error {stock}: {e}")
            continue

    print("\nMatching Stocks:")
    print(results)

    pd.Series(results).to_csv("data/signals.csv", index=False)

if __name__ == "__main__":
    run_scanner()
