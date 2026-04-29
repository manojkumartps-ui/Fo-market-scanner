import pandas as pd
import numpy as np

FILE = "data/nse_fno_ohlc.csv"

# --- YOUR ORIGINAL LOGIC (Heikin Ashi, EMA, check_stock) REMAINS UNTOUCHED ---
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

def run_scanner():
    try:
        # Step 1: Align with your CSV structure (Price/Ticker/Metric)
        # Using header=[1, 2] ignores the 'Price' label and sets up MultiIndex correctly
        raw = pd.read_csv(FILE, header=[1, 2], index_col=0)
        
        # Step 2: Clean dates and remove the 'Date' row shown in your image
        raw.index = pd.to_datetime(raw.index, format="%d-%m-%Y", errors="coerce")
        raw = raw[raw.index.notna()].sort_index()

        results = []
        tickers = raw.columns.get_level_values(0).unique()

        # --- DEBUG WINDOW FOR FIRST STOCK ---
        if len(tickers) > 0:
            test_stock = tickers[0]
            print(f"\n--- DEBUG WINDOW: Checking {test_stock} ---")
            test_df = raw[test_stock].copy()
            test_df.columns = [str(c).strip().capitalize() for c in test_df.columns]
            print(f"Columns Found: {test_df.columns.tolist()}")
            test_df = test_df.apply(pd.to_numeric, errors='coerce').dropna()
            print(f"Data Rows after cleaning: {len(test_df)}")
            if len(test_df) > 0:
                print(f"Sample Data:\n{test_df.tail(2)}")
            print("---------------------------------------\n")

        for stock in tickers:
            if "Unnamed" in str(stock) or not stock: continue
            try:
                df = raw[stock].copy()
                df.columns = [str(c).strip().capitalize() for c in df.columns]
                df = df[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors='coerce').dropna()

                if len(df) < 30: continue
                
                df = pd.concat([df, heikin_ashi(df)], axis=1)
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
