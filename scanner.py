import requests
import zipfile
import io
import os
import pandas as pd
from datetime import datetime, timedelta

FILE = "data/nse_fno_ohlc.csv"


def download_bhavcopy(date):
    date_str = date.strftime("%d%b%Y").upper()
    url = f"https://nsearchives.nseindia.com/content/fo/fo{date_str}bhav.csv.zip"

    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=10)

        if r.status_code != 200:
            print(f"❌ {date_str} not available")
            return None

        z = zipfile.ZipFile(io.BytesIO(r.content))
        file_name = z.namelist()[0]

        df = pd.read_csv(z.open(file_name))
        df["DATE"] = pd.to_datetime(date)

        print(f"✅ {date_str}")
        return df

    except Exception as e:
        print(f"❌ Error {date_str}:", e)
        return None


def main():
    os.makedirs("data", exist_ok=True)

    # ✅ Load existing file
    if os.path.exists(FILE):
        old = pd.read_csv(FILE, parse_dates=["DATE"])
        old = old.sort_values("DATE")

        last_date = old.iloc[-1]["DATE"].date()
        print("Last updated:", last_date)
    else:
        old = pd.DataFrame()
        last_date = datetime(2020, 1, 1).date()
        print("No file found → starting fresh")

    today = datetime.now().date()
    current = last_date + timedelta(days=1)

    new_data = []

    # ✅ Fill missing days
    while current <= today:
        df = download_bhavcopy(current)
        if df is not None:
            new_data.append(df)

        current += timedelta(days=1)

    if not new_data:
        print("⚠️ No new updates")
        return

    new = pd.concat(new_data, ignore_index=True)

    # ✅ Merge
    final = pd.concat([old, new], ignore_index=True)

    # ✅ Clean + sort
    final = final.drop_duplicates()
    final = final.sort_values("DATE")

    # ✅ Save
    final.to_csv(FILE, index=False)

    print("✅ File updated till:", today)


if __name__ == "__main__":
    main()
