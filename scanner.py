import pandas as pd
import numpy as np

FILE = "data/nse_fno_ohlc.csv"

# -----------------------------
# Heikin Ashi - Original Logic
# -----------------------------
def heikin_ashi(df):
    ha = pd.DataFrame(index=df.index)
    ha["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    
    # Calculate HA_Open recursively
    ha_open = [(df["Open"].iloc[0] + df["Close"].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha["HA_Close"].iloc[i - 1]) / 2)
    
    ha["HA_Open"] = ha_open
    return ha

# -----------------------------
# EMA - Original Logic
# -----------------------------
def add_ema(df):
    df["EMA_HA_Open"] = df["HA_Open"].ewm(span=5, adjust=False).mean()
    df["EMA_HA_Close"] = df["HA_Close"].ewm(span=5, adjust=False).mean()
    return df

# -----------------------------
# Strategy - Original Logic + Safety
# -----------------------------
def check_stock(df):
    if len(df) < 30:
        return False

    # 1. Prepare Daily Data (Already has indicators from loop)
    latest = df.iloc[-1]
    d_3 = df.iloc[-4] if len(df) >= 4 else None

    # 2. Prepare Weekly Data
    weekly = df.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last"
    }).dropna()

    if len(weekly) < 5:
        return False

    ha_w = heikin_ashi(weekly)
    weekly = pd.concat([weekly, ha_w], axis=1)
    weekly = add_ema(weekly)

    w_latest = weekly.iloc[-1]
    w_3 = weekly.iloc[-4] if len(weekly) >= 4 else None

    # 3. Apply your combined Boolean logic
    if d_3 is None or w_3 is None:
        return False

    return (
        # Daily Conditions
        latest["Close"] > latest["Open"] and
        latest["Close"] > latest["EMA_HA_Close"] and
        latest["EMA_HA_Open"] < latest["EMA_HA_Close"] and
        d_3["EMA_HA_Open"] > d_3["EMA_HA_Close"] and
        
        # Weekly Conditions
        w_latest["Close"] > w_latest["Open"] and
        w_latest["Close"] > w_latest["EMA_HA_Close"] and
        w_latest["EMA_HA_Open"] < w_latest["EMA_HA_Close"] and
        w_3["EMA_HA_Open"] > w_3["EMA_HA_Close"]
    )

# -----------------------------
# Main Scanner - Integrated Parsing
# -----------------------------
def run_scanner():
    try:
        # Integrated fix: Read Multi-row header (Row 1=Ticker, Row 2=Metric)
        raw = pd.read_csv(FILE, header=[0, 1], index_col=0)
        
        # Date Cleanup (handles the "Date" row and corruption from Excel)
        raw.index = pd.to_datetime(raw.index, errors="coerce", dayfirst=True)
        raw = raw[raw.index.notna()].sort_index()
        
        results = []
        # Tickers are at the top level of the columns
        tickers = raw.columns.get_level_values(0).unique()

        for stock in tickers:
            try:
                # Select only the 4 OHLC columns for this ticker
                df = raw[stock].copy()
                df = df[["Open", "High", "Low", "Close"]]
                df = df.apply(pd.to_numeric, errors='coerce').dropna()

                if len(df) < 30:
                    continue

                # Add Indicators
                ha = heikin_ashi(df)
                df = pd.concat([df, ha], axis=1)
                df = add_ema(df)

                # Run integrated strategy
                if check_stock(df):
                    results.append(stock)

            except Exception:
                continue

        print(f"\nMatching Stocks: {results}")
        pd.Series(results).to_csv("data/signals.csv", index=False)

    except Exception as e:
        print(f"File Error: {e}")

if __name__ == "__main__":
    run_scanner()
