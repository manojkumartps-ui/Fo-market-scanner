import pandas as pd

FILE = "data/nse_fno_ohlc.csv"

# -----------------------------
# Heikin Ashi Calculation
# -----------------------------
def heikin_ashi(df):
    ha = pd.DataFrame(index=df.index)

    # HA Close
    ha["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4

    # HA Open (iterative)
    ha_open = [(df["Open"].iloc[0] + df["Close"].iloc[0]) / 2]

    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha["HA_Close"].iloc[i-1]) / 2)

    ha["HA_Open"] = ha_open

    return ha


# -----------------------------
# EMA Calculation
# -----------------------------
def add_ema(df):
    df["EMA_HA_Open"] = df["HA_Open"].ewm(span=5, adjust=False).mean()
    df["EMA_HA_Close"] = df["HA_Close"].ewm(span=5, adjust=False).mean()
    return df


# -----------------------------
# Strategy Conditions
# -----------------------------
def check_stock(df):

    # Need enough data
    if len(df) < 30:
        return False

    # --- Daily ---
    latest = df.iloc[-1]
    d_3 = df.iloc[-4]

    # --- Weekly ---
    weekly = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()

    ha_w = heikin_ashi(weekly)
    weekly = pd.concat([weekly, ha_w], axis=1)
    weekly = add_ema(weekly)

    if len(weekly) < 5:
        return False

    w_latest = weekly.iloc[-1]
    w_3 = weekly.iloc[-4]

    # --- Conditions ---
    return (
        # Daily
        latest["Close"] > latest["Open"] and
        latest["Close"] > latest["EMA_HA_Close"] and
        latest["EMA_HA_Open"] < latest["EMA_HA_Close"] and

        # 3 days ago (bearish)
        d_3["EMA_HA_Open"] > d_3["EMA_HA_Close"] and

        # Weekly
        w_latest["Close"] > w_latest["Open"] and
        w_latest["Close"] > w_latest["EMA_HA_Close"] and
        w_latest["EMA_HA_Open"] < w_latest["EMA_HA_Close"] and

        # 3 weeks ago (bearish)
        w_3["EMA_HA_Open"] > w_3["EMA_HA_Close"]
    )


# -----------------------------
# Main Scanner
# -----------------------------
def run_scanner():

    data = pd.read_csv(FILE, header=[0, 1], index_col=0, parse_dates=True)

    results = []

    for stock in data.columns.levels[0]:
        try:
            df = data[stock].dropna().copy()

            # Add HA + EMA
            ha = heikin_ashi(df)
            df = pd.concat([df, ha], axis=1)
            df = add_ema(df)

            if check_stock(df):
                results.append(stock)

        except Exception as e:
            print(f"Error in {stock}: {e}")
            continue

    print("\nMatching Stocks:")
    print(results)

    # Save results
    pd.Series(results).to_csv("data/signals.csv", index=False)


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    run_scanner()
