import pandas as pd
import yfinance as yf
import requests
import os
import time

FILE = "data/nse_fno_ohlc.csv"


def get_fno():
    url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/"
    }

    try:
        session = requests.Session()

        # warm-up request (needed for cookies)
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        r = session.get(url, headers=headers, timeout=10)
        r.raise_for_status()

        data = r.json()
        stocks = [x["symbol"] + ".NS" for x in data.get("data", [])]

        print("FNO stocks fetched:", len(stocks))
        return stocks

    except Exception as e:
        print("❌ NSE fetch failed:", e)
        return []


def fetch(stocks):
    if not stocks:
        print("❌ No stocks received for fetch()")
        return pd.DataFrame()

    try:
        df = yf.download(
            stocks,
            period="10d",
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False
        )

        if df.empty:
            print("❌ yfinance returned empty dataframe")
            return pd.DataFrame()

        out = {}

        for s in stocks:
            try:
                if s in df.columns.levels[0]:
                    temp = df[s][["Open", "High", "Low", "Close"]].dropna()
                    if not temp.empty:
                        out[s] = temp
            except Exception:
                continue

        result = pd.concat(out, axis=1) if out else pd.DataFrame()

        print("Fetched data shape:", result.shape)
        return result

    except Exception as e:
        print("❌ yfinance error:", e)
        return pd.DataFrame()


def main():
    old = (
        pd.read_csv(FILE, header=[0, 1], index_col=0, parse_dates=True)
        if os.path.exists(FILE)
        else pd.DataFrame()
    )

    stocks = get_fno()

    if not stocks:
        print("❌ No FNO stocks fetched → stopping")
        return

    new = fetch(stocks)

    if new.empty:
        print("❌ No new OHLC data → stopping")
        return

    if not old.empty:
        final = old.combine_first(new)
        final.update(new)
    else:
        final = new

    final = final.sort_index()

    os.makedirs("data", exist_ok=True)
    final.to_csv(FILE)

    print("✅ Data updated successfully")


if __name__ == "__main__":
    main()
