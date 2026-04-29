import pandas as pd
import numpy as np

FILE = "data/nse_fno_ohlc.csv"

# --- YOUR ORIGINAL LOGIC (Heikin Ashi & EMA) ---
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

# --- STRATEGY WITH ONE MASKED CONDITION ---
def check_stock(df, stock_name):
    if len(df) < 30: return False
    latest = df.iloc[-1]
    d_3 = df.iloc[-4]
    
    weekly = df.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last"
    }).dropna()
    
    if len(weekly) < 5: return False
    
    ha_w = heikin_ashi(weekly)
    weekly = pd.concat([weekly, ha_w], axis=1)
    weekly = add_ema(weekly)
    
    w_latest = weekly.iloc[-1]
    w_3 = weekly.iloc[-4]

    # I have commented out the 4th condition (Daily Lookback) to see if we get hits
    conditions = [
        latest["Close"] > latest["Open"],
        latest["Close"] > latest["EMA_HA_Close"],
        latest["EMA_HA_Open"] < latest["EMA_HA_Close"],
        # d_3["EMA_HA_Open"] > d_3["EMA_HA_Close"],  # <--- MASKED
        w_latest["Close"] > w_latest["Open"],
        w_latest["Close"] > w_latest["EMA_HA_Close"],
        w_latest["EMA_HA_Open"] < w_latest["EMA_HA_Close"],
        w_3["EMA_HA_Open"] > w_3["EMA_HA_Close"]
    ]

    return all(conditions)

def run_scanner():
    try:
        # Fixed the headers to align with your Ticker/Price layout
        raw = pd.read_csv(FILE, header=[1, 2], index_col=0)
        raw.index = pd.to_datetime(raw.index, errors="coerce", dayfirst=True)
        raw = raw[raw.index.notna()].sort_index()
        
        # Ensure Ticker is Level 0
        if "Open" in raw.columns.levels[0]:
            raw.columns = raw.columns.swaplevel(0, 1)
        
        tickers = raw.columns.get_level_values(0).unique()
        results = []

        for stock in tickers:
            if "Unnamed" in str(stock) or not stock: continue
            try:
                df = raw[stock].copy()
                df.columns = [str(c).strip().capitalize() for c in df.columns]
                df = df[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors='coerce').dropna()
                
                if len(df) < 30: continue
                
                df = pd.concat([df, heikin_ashi(df)], axis=1)
                df = add_ema(df)

                if check_stock(df, stock):
                    results.append(stock)
            except: continue

        print(f"\nFINAL Matching Stocks (Daily Lookback Masked): {results}")
        pd.Series(results).to_csv("data/signals.csv", index=False)

    except Exception as e:
        print(f"Scanner Error: {e}")

if __name__ == "__main__":
    run_scanner()
