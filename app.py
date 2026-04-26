import pandas as pd
import yfinance as yf
import requests
from io import StringIO

# Step 1: Get F&O stock list from NSE
url = "https://www.nseindia.com/content/fo/fo_mktlots.csv"

headers = {
    "User-Agent": "Mozilla/5.0"
}

res = requests.get(url, headers=headers)
csv_data = StringIO(res.text)

df_lots = pd.read_csv(csv_data)

# Clean symbol list
symbols = df_lots['SYMBOL'].dropna().unique()

# Convert to Yahoo format
stocks = [s + ".NS" for s in symbols]

print(f"Total F&O stocks: {len(stocks)}")

# Step 2: Fetch OHLC
all_data = {}

for stock in stocks:
    try:
        df = yf.download(stock, period="5mo", interval="1d", progress=False)
        if not df.empty:
            all_data[stock] = df[['Open', 'High', 'Low', 'Close']]
            print(stock, "done")
    except Exception as e:
        print(stock, "failed")

# Step 3: Save
final_df = pd.concat(all_data, axis=1)
final_df.to_csv("nse_fno_ohlc_5months.csv")

print("All done ✅")
