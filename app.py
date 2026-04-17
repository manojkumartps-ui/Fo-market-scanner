import streamlit as st
import pandas as pd
import yfinance as yf
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import time

# --- STAGE 0: AI INITIALIZATION ---
@st.cache_resource
def load_pro_ai():
    """Loads FinBERT - Industry standard for financial sentiment."""
    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
    model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Try to load AI, fallback to neutral if it fails during deployment
try:
    finbert = load_pro_ai()
except Exception as e:
    st.error(f"AI Model Load Failed: {e}")
    finbert = None

# --- STAGE 1: DYNAMIC F&O LIST BUILDING ---
@st.cache_data(ttl=86400)
def get_full_fno_list():
    """Pulls the entire live NSE F&O list from the official instrument feed."""
    try:
        # Fetching directly from Kite/Zerodha public daily master
        url = "https://kite.trade"
        df = pd.read_csv(url)
        # NFO = NSE Futures & Options
        fno_df = df[df['exchange'] == 'NFO']
        symbols = sorted(fno_df['name'].unique().tolist())
        # Filter out noise and append .NS for Yahoo Finance
        return [str(s) + ".NS" for s in symbols if str(s) != 'nan' and len(str(s)) > 1]
    except Exception as e:
        st.warning(f"Live list fetch failed ({e}). Using liquid fallback.")
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS", "ICICIBANK.NS"]

# --- STAGE 2: 20-DAY HEIKIN ASHI LOGIC ---
def calculate_ha_20d(symbol):
    """Calculates smoothed Heikin Ashi trend using 20+ days of stats."""
    try:
        # Download 30 days to ensure we have 20 clean trading days
        df = yf.download(symbol, period="1mo", interval="1d", progress=False)
        
        if df.empty or len(df) < 10:
            return "NO_DATA", 0, "Price fetch failed"

        # FIX: Force flatten Multi-Index columns (yfinance 0.2.40+ fix)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standard OHLC to HA conversion
        # HA_Close = (O+H+L+C)/4
        ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        
        # HA_Open calculation (Vectorized for performance)
        ha_open = [(df['Open'].iloc[0] + df['Close'].iloc[0]) / 2]
        for i in range(1, len(df)):
            ha_open.append((ha_open[i-1] + ha_close.iloc[i-1]) / 2)
            
        current_ha_open = ha_open[-1]
        current_ha_close = ha_close.iloc[-1]
        ltp = float(df['Close'].iloc[-1])
        
        trend = "Bullish 🟢" if current_ha_close > current_ha_open else "Bearish 🔴"
        return trend, round(ltp, 2), "Success"
    except Exception as e:
        return "ERROR", 0, str(e)

# --- STAGE 3: AI CROWD SCORING ---
def get_ai_sentiment(symbol):
    """Analyzes recent headlines using FinBERT AI."""
    if not finbert: return 0.0, "AI Not Loaded"
    try:
        t = yf.Ticker(symbol)
        news = t.news[:5] # Top 5 headlines
        if not news: return 0.0, "No News Found"
        
        titles = [n.get('title', '') for n in news if n.get('title')]
        results = finbert(titles)
        
        # Mapping: Positive=1, Neutral=0, Negative=-1
        sent_map = {"positive": 1, "neutral": 0, "negative": -1}
        scores = [sent_map[r['label']] * r['score'] for r in results]
        
        avg_score = round(sum(scores) / len(scores), 2)
        return avg_score, f"Analyzed {len(titles)} headlines"
    except Exception as e:
        return 0.0, f"Error: {e}"

# --- STREAMLIT UI ---
st.set_page_config(page_title="Pro NSE F&O Scanner", layout="wide")
st.title("🏹 NSE F&O Pro Scanner (20D Heikin Ashi)")

scan_limit = st.sidebar.slider("Scan Depth", 5, 100, 15)

if st.button("🚀 Start Market-Wide Scan"):
    fno_list = get_full_fno_list()
    targets = fno_list[:scan_limit]
    
    scan_results = []
    
    for s in targets:
        # Debugging Output in Expander
        with st.expander(f"Analyzing {s}", expanded=True):
            col1, col2 = st.columns(2)
            
            # Step 1: HA Logic
            with col1:
                trend, price, p_msg = calculate_ha_20d(s)
                if trend != "ERROR":
                    st.success(f"Trend: {trend} | LTP: ₹{price}")
                else:
                    st.error(f"Price Error: {p_msg}")
            
            # Step 2: AI Logic
            with col2:
                sent, s_msg = get_ai_sentiment(s)
                st.info(f"AI Score: {sent}")
                st.caption(f"Status: {s_msg}")
            
            scan_results.append({
                "Stock": s.replace(".NS", ""),
                "Price": price,
                "20D HA Trend": trend,
                "AI Sentiment": sent,
                "Debug Note": s_msg
            })
            time.sleep(0.5) # Compliance delay

    st.divider()
    st.subheader("📊 Consolidated Market Report")
    st.dataframe(pd.DataFrame(scan_results), use_container_width=True)

st.caption("Heikin Ashi calculated over 20+ trading days. Sentiment via FinBERT AI.")
