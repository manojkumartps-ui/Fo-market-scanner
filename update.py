import pandas as pd
import requests
import os
import gzip
import io
from datetime import datetime, timedelta

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
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        r = session.get(url, headers=headers, timeout=10)
        r.raise_for_status()

        data = r.json()
        return [x["symbol"] + ".NS" for x in data.get("data", [])]

    except Exception as e:
        print("❌ NSE fetch failed:", e)
        return []


def fetch(stocks):
    def get_bhavcopy(date):
        date_str = date.strftime("%d%m%Y")
        url = f"https://archives.nseindia.com/content/fo/fo_{date_str}.csv.gz"

        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if r.status_code != 200:
                return None

            with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as f:
                return pd.read_csv(f)
        except:
            return None

    # ✅ FIXED DATE HANDLING
    if os.path.exists(FILE):
        old = pd.read_csv(FILE, header=[0, 1], index_col=0)

        # 🔥 FORCE PROPER DATE PARSING (CRITICAL FIX)
        old.index = pd.to_datetime(old.index, format="%d-%m-%Y", errors="coerce")
        old = old.dropna()

        if old.empty:
            start = datetime.now() - timedelta(days=5)
        else:
            start = old.index.max() + timedelta(days=1)
    else:
        start = datetime.now() - timedelta(days=5)

    end = datetime.now() - timedelta(days=1)  # avoid incomplete today

    print(f"Fetching from {start.date()} to {end.date()}")

    all_days = []
    cur = start

    while cur <= end:
        df = get_bhavcopy(cur)

        if df is not None:
            df = df[df["INSTRUMENT"] == "FUTSTK"]

            day_data = {}

            for _, row in df.iterrows():
                s = row["SYMBOL"] + ".NS"
                if stocks and s not in stocks:
                    continue

                day_data.setdefault(s, {})[cur] = {
                    "Open": row["OPEN"],
                    "High": row["HIGH"],
                    "Low": row["LOW"],
                    "Close": row["CLOSE"]
                }

            if day_data:
                day_df = pd.concat(
                    {k: pd.DataFrame(v).T for k, v in day_data.items()},
                    axis=1
                )
                all_days.append(day_df)

        cur += timedelta(days=1)

    if not all_days:
        return pd.DataFrame()

    return pd.concat(all_days).sort_index()


def main():
    if os.path.exists(FILE):
        old = pd.read_csv(FILE, header=[0, 1], index_col=0)

        # 🔥 CRITICAL FIX
        old.index = pd.to_datetime(old.index, format="%d-%m-%Y", errors="coerce")
        old = old.dropna()
    else:
        old = pd.DataFrame()

    stocks = get_fno()
    new = fetch(stocks)

    if new.empty:
        print("❌ No new data")
        return

    if not old.empty:
        final = old.combine_first(new)
        final.update(new)
    else:
        final = new

    final = final.sort_index()

    os.makedirs("data", exist_ok=True)
    final.to_csv(FILE)

    print("✅ Updated successfully")


if __name__ == "__main__":
    main()
