import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import time

st.title("📊 NSE F&O OHLC Downloader")

# -----------------------------
# STEP 1: Get F&O stocks
# -----------------------------
def get_fno_stocks():
    try:
        session = requests.Session()

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/"
        }

        session.get("https://www.nseindia.com", headers=headers, timeout=5)

        url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
        res = session.get(url, headers=headers, timeout=5)

        data = res.json()
        symbols = [item['symbol'] for item in data['data']]

        return [s + ".NS" for s in symbols]

    except:
        st.warning("NSE blocked request → using fallback list")
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]


# -----------------------------
# STEP 2: Fetch OHLC
# -----------------------------
def fetch_data(stocks):
    all_data = {}

    progress = st.progress(0)

    for i, stock in enumerate(stocks):
        try:
            df = yf.download(stock, period="5mo", interval="1d", progress=False)

            if not df.empty:
                all_data[stock] = df[['Open', 'High', 'Low', 'Close']]

        except:
            pass

        progress.progress((i + 1) / len(stocks))

        time.sleep(0.1)

    return all_data


# -----------------------------
# RUN BUTTON
# -----------------------------
if st.button("🚀 Run Fetch"):

    stocks = get_fno_stocks()

    st.write(f"Total F&O Stocks: {len(stocks)}")

    data = fetch_data(stocks)

    if len(data) == 0:
        st.error("No data fetched")
        st.stop()

    final_df = pd.concat(data, axis=1)

    st.success("Data ready!")

    st.dataframe(final_df.head())

    # -----------------------------
    # DOWNLOAD BUTTON (IMPORTANT)
    # -----------------------------
    csv = final_df.to_csv().encode('utf-8')

    st.download_button(
        label="📥 Download OHLC CSV",
        data=csv,
        file_name="fno_ohlc_5months.csv",
        mime="text/csv"
    )
