import streamlit as st
import pandas as pd

from generators.ydata_generator import generate_ydata
from utils.preprocess import clean_dataframe
from utils.exporters import download_buttons

st.set_page_config(page_title="YData Synthetic Lab", layout="wide")

st.title("YData Synthetic Data Generator")

rows = st.sidebar.slider("Synthetic Rows", 100, 50000, 1000)
epochs = st.sidebar.slider("Training Epochs", 100, 800, 400)

uploaded = st.file_uploader("Upload CSV dataset (recommended 500+ rows)")

df = None

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Real Data Preview")
    st.dataframe(df.head())

if st.button("Generate Synthetic Data"):

    if df is None:
        st.error("Upload dataset first")
        st.stop()

    if len(df) < 200:
        st.error("Need at least 200 rows for realistic YData synthesis")
        st.stop()

    df_clean = clean_dataframe(df)

    with st.spinner("Training YData model — please wait..."):
        synth = generate_ydata(df_clean, rows, epochs)

    st.subheader("Synthetic Data Preview")
    st.dataframe(synth.head())

    download_buttons(synth)
