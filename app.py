import requests
import pandas as pd
import yfinance as yf
import time

# -----------------------------
# STEP 1: Get F&O stock list
# -----------------------------
def get_fno_stocks():
    try:
        session = requests.Session()

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/"
        }

        # जरूरी: cookie set करना
        session.get("https://www.nseindia.com", headers=headers, timeout=5)

        url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
        res = session.get(url, headers=headers, timeout=5)

        data = res.json()

        symbols = [item['symbol'] for item in data['data']]

        stocks = [s + ".NS" for s in symbols]

        print(f"Fetched {len(stocks)} F&O stocks")

        return stocks

    except Exception as e:
        print("NSE failed, using fallback list ❌")

        # fallback (so app never crashes)
        return [
            "RELIANCE.NS", "TCS.NS", "INFY.NS",
            "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"
        ]


# -----------------------------
# STEP 2: Fetch OHLC
# -----------------------------
def fetch_ohlc(stocks):
    all_data = {}

    for stock in stocks:
        try:
            df = yf.download(
                stock,
                period="5mo",
                interval="1d",
                progress=False
            )

            if not df.empty:
                all_data[stock] = df[['Open', 'High', 'Low', 'Close']]
                print(f"{stock} ✅")

            time.sleep(0.2)  # avoid rate limit

        except Exception as e:
            print(f"{stock} ❌")

    return all_data


# -----------------------------
# STEP 3: Run everything
# -----------------------------
def main():
    stocks = get_fno_stocks()

    data = fetch_ohlc(stocks)

    if len(data) == 0:
        print("No data fetched ❌")
        return

    final_df = pd.concat(data, axis=1)

    final_df.to_csv("fno_ohlc_5months.csv")

    print("DONE 🚀 File saved: fno_ohlc_5months.csv")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()
