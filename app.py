import streamlit as st
import pandas as pd
import yfinance as yf
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import time

# --- STAGE 0: ADVANCED AI LOAD ---
@st.cache_resource
def load_pro_ai():
    # FinBERT is the industry standard for financial sentiment
    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
    model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

finbert = load_pro_ai()

# --- STAGE 1: COMPLETE F&O LIST BUILDING ---
@st.cache_data(ttl=86400)
def get_full_fno_list():
    try:
        # Fetch entire daily instrument master from Zerodha/Kite
        df = pd.read_csv("https://api.kite.trade/instruments")
        # Filter strictly for NSE Derivatives (NFO) segment
        fno_df = df[df['exchange'] == 'NFO']
        # Extract unique underlying stock symbols
        symbols = sorted(fno_df['name'].unique().tolist())
        return [str(s) + ".NS" for s in symbols if str(s) != 'nan']
    except Exception as e:
        st.error(f"Failed to fetch live NSE list: {e}")
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"] # Minimal fallback

# --- STAGE 2: 20-DAY HEIKIN ASHI CALCULATION ---
def calculate_heikin_ashi_20d(symbol):
    try:
        # 20-day history for accurate smoothing
        df = yf.download(symbol, period="25d", interval="1d", progress=False, multi_level_index=False)
        if df.empty or len(df) < 20: return "ERROR", 0
        
        # Initialize HA columns
        ha_df = pd.DataFrame(index=df.index)
        ha_df['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        
        # Seed first HA Open
        ha_df.at[df.index[0], 'Open'] = (df.iloc[0]['Open'] + df.iloc[0]['Close']) / 2
        
        # Vectorized calculation for remaining days
        for i in range(1, len(df)):
            ha_df.at[df.index[i], 'Open'] = (ha_df.at[df.index[i-1], 'Open'] + ha_df.at[df.index[i-1], 'Close']) / 2
            
        # Determine final trend from the last 20 candles
        curr_ha_open = ha_df['Open'].iloc[-1]
        curr_ha_close = ha_df['Close'].iloc[-1]
        
        trend = "Bullish 🟢" if curr_ha_close > curr_ha_open else "Bearish 🔴"
        return trend, round(float(df['Close'].iloc[-1]), 2)
    except:
        return "ERROR", 0

# --- STAGE 3: LIVE CROWD SCORING ---
def get_ai_sentiment(symbol):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news: return 0.0
        titles = [n.get('title', '') for n in news[:5]]
        results = finbert(titles)
        # Weighting: Positive=1, Neutral=0, Negative=-1
        sent_map = {"positive": 1, "neutral": 0, "negative": -1}
        scores = [sent_map[r['label']] * r['score'] for r in results]
        return round(sum(scores) / len(scores), 2)
    except:
        return 0.0

# --- UI INTERFACE ---
st.title("🛡️ Full-Market NSE F&O Scanner (20D Heikin Ashi)")
scan_limit = st.sidebar.slider("Number of stocks to scan", 10, 190, 20)

if st.button("🚀 Execute Market-Wide Scan"):
    fno_list = get_full_fno_list()
    scan_targets = fno_list[:scan_limit]
    
    results = []
    prog = st.progress(0)
    
    for i, s in enumerate(scan_targets):
        trend, price = calculate_heikin_ashi_20d(s)
        score = get_ai_sentiment(s)
        
        results.append({
            "Stock": s.replace(".NS", ""),
            "LTP": price,
            "20D HA Trend": trend,
            "AI Sentiment": score
        })
        prog.progress((i + 1) / len(scan_targets))
    
    st.table(pd.DataFrame(results))
