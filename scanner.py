import pandas as pd
import numpy as np

FILE = "data/nse_fno_ohlc.csv"

# -----------------------------
# CORE INDICATORS (UNCHANGED)
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

# -----------------------------
# CE STRATEGY (BULLISH REVERSAL)
# -----------------------------
def is_ce_candidate(df):
    if len(df) < 30: return False
    latest = df.iloc[-1]
    d_3 = df.iloc[-4]
    
    weekly = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    if len(weekly) < 5: return False
    
    ha_w = heikin_ashi(weekly)
    weekly = pd.concat([weekly, ha_w], axis=1); weekly = add_ema(weekly)
    w_latest = weekly.iloc[-1]; w_3 = weekly.iloc[-4]

    return (
        latest["Close"] > latest["Open"] and
        latest["Close"] > latest["EMA_HA_Close"] and
        latest["EMA_HA_Open"] < latest["EMA_HA_Close"] and
        d_3["EMA_HA_Open"] > d_3["EMA_HA_Close"] and # Was Bearish 4 days ago
        w_latest["Close"] > w_latest["Open"] and
        w_latest["Close"] > w_latest["EMA_HA_Close"] and
        w_latest["EMA_HA_Open"] < w_latest["EMA_HA_Close"] and
        w_3["EMA_HA_Open"] > w_3["EMA_HA_Close"]     # Was Bearish 4 weeks ago
    )

# -----------------------------
# PE STRATEGY (BEARISH REVERSAL)
# -----------------------------
def is_pe_candidate(df):
    if len(df) < 30: return False
    latest = df.iloc[-1]
    d_3 = df.iloc[-4]
    
    weekly = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    if len(weekly) < 5: return False
    
    ha_w = heikin_ashi(weekly)
    weekly = pd.concat([weekly, ha_w], axis=1); weekly = add_ema(weekly)
    w_latest = weekly.iloc[-1]; w_3 = weekly.iloc[-4]

    return (
        latest["Close"] < latest["Open"] and          # Red Candle
        latest["Close"] < latest["EMA_HA_Close"] and  # Below EMA
        latest["EMA_HA_Open"] > latest["EMA_HA_Close"] and # Bearish Trend
        d_3["EMA_HA_Open"] < d_3["EMA_HA_Close"] and  # Was Bullish 4 days ago
        w_latest["Close"] < w_latest["Open"] and
        w_latest["Close"] < w_latest["EMA_HA_Close"] and
        w_latest["EMA_HA_Open"] > w_latest["EMA_HA_Close"] and
        w_3["EMA_HA_Open"] < w_3["EMA_HA_Close"]      # Was Bullish 4 weeks ago
    )

# -----------------------------
# MAIN SCANNER
# -----------------------------
def run_scanner():
    try:
        raw = pd.read_csv(FILE, header=[1, 2], index_col=0)
        raw.index = pd.to_datetime(raw.index, errors="coerce", dayfirst=True)
        raw = raw[raw.index.notna()].sort_index()
        if "Open" in raw.columns.levels[0]: raw.columns = raw.columns.swaplevel(0, 1)
        
        tickers = raw.columns.get_level_values(0).unique()
        ce_results, pe_results = [], []

        for stock in tickers:
            if "Unnamed" in str(stock) or not stock: continue
            try:
                df = raw[stock].copy()
                df.columns = [str(c).strip().capitalize() for c in df.columns]
                df = df[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors='coerce').dropna()
                if len(df) < 30: continue
                
                df = pd.concat([df, heikin_ashi(df)], axis=1); df = add_ema(df)

                if is_ce_candidate(df): ce_results.append(stock)
                if is_pe_candidate(df): pe_results.append(stock)
            except: continue

        # --- OUTPUT WINDOWS ---
        print("\n" + "="*40)
        print("     SCANNER RESULTS - CE WINDOW (Bullish)")
        print("="*40)
        if ce_results:
            for s in ce_results: print(f" [CE] -> {s}")
        else:
            print(" No bullish reversals found.")

        print("\n" + "="*40)
        print("     SCANNER RESULTS - PE WINDOW (Bearish)")
        print("="*40)
        if pe_results:
            for s in pe_results: print(f" [PE] -> {s}")
        else:
            print(" No bearish reversals found.")
        print("="*40 + "\n")
        
        # Save to CSV
        output = pd.DataFrame({"Signal": ["CE"]*len(ce_results) + ["PE"]*len(pe_results),
                               "Ticker": ce_results + pe_results})
        output.to_csv("data/signals.csv", index=False)

    except Exception as e:
        print(f"Scanner Error: {e}")

if __name__ == "__main__":
    run_scanner()
