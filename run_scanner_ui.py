import streamlit as st
import pandas as pd
import subprocess
import sys
import os

SIGNALS_FILE = "data/signals.csv"

st.title("Scanner Test Panel (Safe Mode)")

st.info("This does NOT affect historical data or your main app.py")

# -------------------------
# Run Scanner
# -------------------------
if st.button("Run Scanner"):
    st.write("Executing scanner.py...")

    result = subprocess.run(
        [sys.executable, "scanner.py"],
        capture_output=True,
        text=True
    )

    st.success("Scanner completed")

    if result.stdout:
        st.text(result.stdout)

    if result.stderr:
        st.text(result.stderr)


# -------------------------
# Show Signals
# -------------------------
st.subheader("Signals Output")

if os.path.exists(SIGNALS_FILE):
    df = pd.read_csv(SIGNALS_FILE)
    st.dataframe(df)
else:
    st.warning("No signals generated yet")
