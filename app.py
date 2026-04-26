import pandas as pd
import yfinance as yf
import requests
import os
import time

FILE = "data/nse_fno_ohlc.csv"


def get_fno():
    s = requests.Session()
    h = {"User-Agent": "Mozilla/5.0"}

    s.get("https://www.nseindia.com", headers=h)
    r = s.get("https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O", headers=h)

    return [x["symbol"] + ".NS" for x in r.json()["data"]]


def fetch(stocks):
    d = {}
    for s in stocks:
        df = yf.download(s, period="7d", interval="1d", progress=False)
        if not df.empty:
            d[s] = df[["Open","High","Low","Close"]]
        time.sleep(0.1)
    return pd.concat(d, axis=1) if d else pd.DataFrame()


old = pd.read_csv(FILE, header=[0,1], index_col=0, parse_dates=True) if os.path.exists(FILE) else pd.DataFrame()

new = fetch(get_fno())

final = pd.concat([old, new]) if not old.empty else new
final = final[~final.index.duplicated(keep="last")].sort_index()

final.to_csv(FILE)
