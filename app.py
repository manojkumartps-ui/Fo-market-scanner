NSE Primary Blocked (Error 403).        
        # Step 2: Fetch the CSV directly from the reports directory
        response = session.get(url_csv, headers=headers, timeout=15)
        
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
            # Standard NSE reports use 'SYMBOL' and ' SERIES'
            # Filtering for 'EQ' ensures we get standard equity stocks
            fno_list = df[df[' SERIES'].str.strip() == 'EQ']['SYMBOL'].unique().tolist()
            return sorted(fno_list)
        else:
            # If still blocked, we acknowledge the server-side restriction
            st.error(f"NSE Primary Blocked (Error {response.status_code}).")
            return []
    except Exception as e:
        st.error(f"Legitimate fetch failed: {e}")
        return []

symbols = get_fno()

# ================= DATA LOAD =================

@st.cache_data
def load(symbols):
    if not symbols: return pd.DataFrame()
    # Scans top 100 for speed on cloud environment
    tickers = [s + ".NS" for s in symbols[:100]] 
    return yf.download(tickers=tickers, period="6mo", interval="1d", group_by="ticker", threads=True, progress=False)

data = load(symbols)

# ================= DUAL SIGNAL ENGINE (SHA + SMC) =================

def evaluate_dual(df):
    df = df.dropna().copy()
    if len(df) < 20: return "NEUTRAL", None

    # HEIKIN ASHI CALC (Your Original)
    ha_close = (df.Open + df.High + df.Low + df.Close) / 4
    ha_open = np.zeros(len(df))
    ha_open[0] = (df.Open.iloc[0] + df.Close.iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
    
    # SMOOTHING
    o2 = pd.Series(ha_open).ewm(span=LEN1, adjust=False).mean().ewm(span=LEN2, adjust=False).mean()
    c2 = pd.Series(ha_close).ewm(span=LEN1, adjust=False).mean().ewm(span=LEN2, adjust=False).mean()
    Hadiff = o2 - c2
    i = len(df) - 1

    # 1. SHA Momentum (Original Logic - Untouched)
    sha_buy = (Hadiff.iloc[i-1] <= 0 and Hadiff.iloc[i] > 0 and df.Close.iloc[i] > df.Open.iloc[i])
    sha_sell = (Hadiff.iloc[i-1] >= 0 and Hadiff.iloc[i] < 0 and df.Close.iloc[i] < df.Open.iloc[i])

    # 2. SMC Structure (BOS/CHoCH Additional Logic)
    swing_high = df['High'].iloc[-SMC_LOOKBACK:-1].max()
    swing_low = df['Low'].iloc[-SMC_LOOKBACK:-1].min()
    
    smc_buy = (df.Close.iloc[i] > swing_high and c2.iloc[i] > o2.iloc[i])
    smc_sell = (df.Close.iloc[i] < swing_low and c2.iloc[i] < o2.iloc[i])

    if sha_buy or smc_buy:
        return "BUY", {"Type": "Momentum" if sha_buy else "SMC BOS"}
    if sha_sell or smc_sell:
        return "SELL", {"Type": "Momentum" if sha_sell else "SMC BOS"}
    
    return "NEUTRAL", None

# ================= RUN =================

if st.button("RUN DUAL SCAN"):
    if not symbols:
        st.error("No symbols loaded. The NSE portal is currently denying automated requests.")
    else:
        buy_list, sell_list = [], []
        for s in symbols[:100]:
            ticker = s + ".NS"
            if ticker not in data or data[ticker].empty: continue
            sig, trace = evaluate_dual(data[ticker])
            if sig == "BUY": buy_list.append(f"{s} ({trace['Type']})")
            elif sig == "SELL": sell_list.append(f"{s} ({trace['Type']})")

        st.subheader(f"🟢 BUY Candidates ({len(buy_list)})")
        st.write(buy_list)
        st.subheader(f"🔴 SELL Candidates ({len(sell_list)})")
        st.write(sell_list)
