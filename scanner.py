import pandas as pd
import numpy as np

FILE = "data/nse_fno_ohlc.csv"

# -----------------------------
# YOUR ORIGINAL LOGIC - UNCHANGED
# -----------------------------
def heikin_ashi(df):
    ha = pd.DataFrame(index=df.index)
    ha["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = [(df["Open"].iloc[0] + df["Close"].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha["HA_Close"].iloc[i - 1]) / 2)
    ha["HA_Open"] = ha_open
    return ha

def add_ema(df):
    df["EMA_HA_Open"] = df["HA_Open"].ewm(span=5, adjust=False).mean()
    df["EMA_HA_Close"] = df["HA_Close"].ewm(span=5, adjust=False).mean()
    return df

def check_stock(df):
    if len(df) < 30: return False
    latest = df.iloc[-1]
    d_3 = df.iloc[-4]
    weekly = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    if len(weekly) < 5: return False
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
# FIXED LOADING FOR YOUR CSV STRUCTURE
# -----------------------------
def run_scanner():
    try:
        # 1. Load CSV by skipping Row 1 ('Price') and using Row 2 (Ticker) & Row 3 (OHLC) as headers
        raw = pd.read_csv(FILE, header=[1, 2], index_col=0, skiprows=[0])

        # 2. Fix the Date Index (Removes the 'Date' label row and #######)
        raw.index = pd.to_datetime(raw.index, errors="coerce", dayfirst=True)
        raw = raw[raw.index.notna()].sort_index()

        results = []
        # Get the unique ticker names from Level 0 of the multi-index columns
        tickers = raw.columns.get_level_values(0).unique()

        for stock in tickers:
            if "Unnamed" in str(stock) or not stock: continue
            
            try:
                # Extract the 4 OHLC columns for the specific ticker
                df = raw[stock].copy()
                
                # Standardize column names for your logic (Open, High, Low, Close)
                df.columns = [str(c).strip().capitalize() for c in df.columns]
                df = df[["Open", "High", "Low", "Close"]]
                
                # Force numeric data and remove gaps
                df = df.apply(pd.to_numeric, errors='coerce').dropna()

                if len(df) < 30: continue

                # Apply your original indicator logic
                ha = heikin_ashi(df)
                df = pd.concat([df, ha], axis=1)
                df = add_ema(df)

                # Check your original strategy logic
                if check_stock(df):
                    results.append(stock)

            except Exception:
                continue

        print("\nMatching Stocks:")
        print(results)
        pd.Series(results).to_csv("data/signals.csv", index=False)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_scanner()
