import pandas as pd
import numpy as np

FILE = "data/nse_fno_ohlc.csv"

# -----------------------------
# YOUR ORIGINAL LOGIC - UNTOUCHED
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
    
    if len(weekly) < 5: return False # <--- POTENTIAL BOTTLENECK
    
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
# SCANNER + DEBUG WINDOW
# -----------------------------
def run_scanner():
    try:
        # Load CSV using Ticker/Price levels
        raw = pd.read_csv(FILE, header=[1, 2], index_col=0)
        raw.index = pd.to_datetime(raw.index, errors="coerce", dayfirst=True)
        raw = raw[raw.index.notna()].sort_index()

        # Fix Swapped Headers
        if "Open" in raw.columns.levels[0]:
            raw.columns = raw.columns.swaplevel(0, 1)
        
        tickers = raw.columns.levels[0].unique()

        # --- DEBUG WINDOW ---
        test_ticker = [t for t in tickers if "Unnamed" not in str(t)][0]
        print(f"\n--- DEBUG WINDOW: {test_ticker} ---")
        test_df = raw[test_ticker].copy()
        test_df.columns = [str(c).strip().capitalize() for c in test_df.columns]
        test_df = test_df[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors='coerce').dropna()
        print(f"Daily Rows: {len(test_df)}")
        
        test_weekly = test_df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
        print(f"Weekly Rows generated: {len(test_weekly)}")
        print(f"Weekly threshold (>=5) met: {len(test_weekly) >= 5}")
        print("-------------------------------\n")

        results = []
        for stock in tickers:
            if "Unnamed" in str(stock) or not stock: continue
            try:
                df = raw[stock].copy()
                df.columns = [str(c).strip().capitalize() for c in df.columns]
                df = df[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors='coerce').dropna()

                if len(df) < 30: continue
                
                ha = heikin_ashi(df)
                df = pd.concat([df, ha], axis=1)
                df = add_ema(df)

                if check_stock(df):
                    results.append(stock)
            except: continue

        print(f"Matching Stocks: {results}")
        pd.Series(results).to_csv("data/signals.csv", index=False)

    except Exception as e:
        print(f"Scanner Error: {e}")

if __name__ == "__main__":
    run_scanner()
