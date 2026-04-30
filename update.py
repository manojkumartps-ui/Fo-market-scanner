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

        # warm-up (required for NSE cookies)
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
        print("❌ No stocks received")
        return pd.DataFrame()

    all_data = {}
    batch_size = 10   # 🔥 IMPORTANT FIX

    for i in range(0, len(stocks), batch_size):
        batch = stocks[i:i + batch_size]

        try:
            df = yf.download(
                batch,
                period="10d",
                interval="1d",
                group_by="ticker",
                threads=False,   # 🔥 prevents Yahoo blocking
                progress=False
            )

            for s in batch:
                try:
                    if s in df.columns.levels[0]:
                        temp = df[s][["Open", "High", "Low", "Close"]].dropna()
                        if not temp.empty:
                            all_data[s] = temp
                except:
                    continue

        except Exception as e:
            print("Batch failed:", batch, e)

        time.sleep(1)  # 🔥 avoid rate limit

    if not all_data:
        print("❌ No OHLC data collected")
        return pd.DataFrame()

    result = pd.concat(all_data, axis=1)
    print("Fetched final shape:", result.shape)
    return result


def main():
    old = (
        pd.read_csv(FILE, header=[0, 1], index_col=0, parse_dates=True)
        if os.path.exists(FILE)
        else pd.DataFrame()
    )

    stocks = get_fno()
    new = fetch(stocks)

    if new.empty:
        print("❌ No new data → exit")
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
