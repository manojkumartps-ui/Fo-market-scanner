import pandas as pd
import numpy as np

FILE = "data/nse_fno_ohlc.csv"

# -----------------------------
# YOUR ORIGINAL HEIKIN ASHI LOGIC
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
# YOUR ORIGINAL EMA LOGIC
# -----------------------------
def add_ema(df):
    df["EMA_HA_Open"] = df["HA_Open"].ewm(span=5, adjust=False).mean()
    df["EMA_HA_Close"] = df["HA_Close"].ewm(span=5, adjust=False).mean()
    return df

# -----------------------------
# YOUR ORIGINAL STRATEGY LOGIC
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
# UPDATED SCANNER (Fixes empty list)
# -----------------------------
def run_scanner():
    try:
        # Load skipping the 'Price' row (0), using Ticker (1) and Metric (2)
        # Based on your image, this is the most reliable way to align headers
        raw = pd.read_csv(FILE, header=[1, 2], index_col=0, skiprows=[0])

        # Clean the date index and remove the "Date" row text seen in your image
        raw.index = pd.to_datetime(raw.index, errors="coerce", dayfirst=True)
        raw = raw[raw.index.notna()].sort_index()

        results = []
        # Get unique tickers from the first level of the MultiIndex columns
        tickers = raw.columns.get_level_values(0).unique()

        for stock in tickers:
            # Skip empty or "Unnamed" ticker names
            if "Unnamed" in stock or not stock:
                continue
                
            try:
                # Extract the 4 columns belonging to this ticker
                df = raw[stock].copy()
                
                # Standardize columns to match your strategy expectations
                df.columns = [c.strip().capitalize() for c in df.columns]
                df = df[["Open", "High", "Low", "Close"]]
                
                # Convert to numbers and drop rows with empty data
                df = df.apply(pd.to_numeric, errors='coerce').dropna()

                if len(df) < 30:
                    continue

                # Apply indicators
                ha = heikin_ashi(df)
                df = pd.concat([df, ha], axis=1)
                df = add_ema(df)

                # Run your strategy
                if check_stock(df):
                    results.append(stock)

            except Exception:
                continue

        print("\nMatching Stocks:")
        print(results)
        pd.Series(results).to_csv("data/signals.csv", index=False)

    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    run_scanner()
