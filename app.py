import streamlit as st
import pandas as pd

st.title("📈 Bank Model Dashboard")

# Load logs
try:
    with open("run_log.txt", "r") as f:
        lines = f.readlines()

    data = [eval(line.strip()) for line in lines if line.strip()]
    df = pd.DataFrame(data)

    st.subheader("Recent Runs")
    st.dataframe(df.tail())

    st.line_chart(df['pnl'])
    st.line_chart(df['accuracy'])

except:
    st.write("No data yet.")