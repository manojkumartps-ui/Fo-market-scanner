import pandas as pd
import numpy as np

FILE = "data/nse_fno_ohlc.csv"

# --- YOUR ORIGINAL LOGIC (UNCHANGED) ---
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

# --- SCANNER WITH DEBUGGING ---
def run_scanner():
    try:
        # Load CSV using rows 1 and 2 as headers (skipping row 0 'Price')
        raw = pd.read_csv(FILE, header=[1, 2], index_col=0)
        
        # Clean Date Index
        raw.index = pd.to_datetime(raw.index, errors="coerce", dayfirst=True)
        raw = raw[raw.index.notna()].sort_index()

        # --- RE-ADDED DEBUG WINDOW ---
        print("\n--- DEBUG: CSV STRUCTURE ---")
        print(f"Column Levels: {raw.columns.names}")
        print(f"Level 0 sample: {list(raw.columns.levels[0])[:3]}")
        print(f"Level 1 sample: {list(raw.columns.levels[1])[:3]}")
        
        # If Level 0 is 'Open/High/Low', we MUST swap them
        if "Open" in raw.columns.levels[0]:
            print("Action: Swapping Levels to put Tickers on top...")
            raw.columns = raw.columns.swaplevel(0, 1)
        
        results = []
        tickers = raw.columns.get_level_values(0).unique()
        print(f"Total Tickers found: {len(tickers)}")

        for stock in tickers:
            if "Unnamed" in str(stock) or not stock: continue
            try:
                # Extract OHLC for the stock
                df = raw[stock].copy()
                
                # Standardize column names
                df.columns = [str(c).strip().capitalize() for c in df.columns]
                df = df[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors='coerce').dropna()

                if len(df) < 30: continue
                
                # Apply indicators and check strategy
                df = pd.concat([df, heikin_ashi(df)], axis=1)
                df = add_ema(df)

                if check_stock(df):
                    results.append(stock)
            except:
                continue

        print(f"\nMatching Stocks: {results}")
        pd.Series(results).to_csv("data/signals.csv", index=False)

    except Exception as e:
        print(f"Scanner Error: {e}")

if __name__ == "__main__":
    run_scanner()
