import streamlit as st


def download_buttons(df):

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "synthetic.csv",
        "text/csv"
    )

    st.download_button(
        "Download JSON",
        df.to_json(orient="records"),
        "synthetic.json"
    )
