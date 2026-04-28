import pandas as pd

FILE = "data/nse_fno_ohlc.csv"

# -----------------------------
# Heikin Ashi
# -----------------------------
def heikin_ashi(df):
    ha = pd.DataFrame(index=df.index)

    ha["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4

    ha_open = [(df["Open"].iloc[0] + df["Close"].iloc[0]) / 2]

    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha["HA_Close"].iloc[i-1]) / 2)

    ha["HA_Open"] = ha_open

    return ha


# -----------------------------
# EMA
# -----------------------------
def add_ema(df):
    df["EMA_HA_Open"] = df["HA_Open"].ewm(span=5, adjust=False).mean()
    df["EMA_HA_Close"] = df["HA_Close"].ewm(span=5, adjust=False).mean()
    return df


# -----------------------------
# Strategy
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

    ha_w = heikin_ashi(weekly)
    weekly = pd.concat([weekly, ha_w], axis=1)
    weekly = add_ema(weekly)

    if len(weekly) < 5:
        return False

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
# FIXED CSV READER (your format)
# -----------------------------
def load_data():

    raw = pd.read_csv(FILE, header=None)

    tickers = raw.iloc[0]
    fields = raw.iloc[1]

    df = raw.iloc[3:].copy()

    cols = []

    for i in range(len(tickers)):
        if i == 0:
            cols.append("Date")
        else:
            cols.append(f"{tickers[i]}_{fields[i]}")

    df.columns = cols

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date")
    df = df.sort_index()

    df = df.apply(pd.to_numeric, errors="coerce")

    return df


# -----------------------------
# MAIN
# -----------------------------
def run_scanner():

    data = load_data()

    results = []

    stocks = set([c.split("_")[0] for c in data.columns if "_" in c])

    for stock in stocks:

        try:
            df = data[[c for c in data.columns if c.startswith(stock + "_")]].copy()
            df.columns = [c.split("_")[1] for c in df.columns]

            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            df = df.sort_index()

            if len(df) < 30:
                continue

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

    pd.Series(results).to_csv("data/signals.csv", index=False)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    run_scanner()
