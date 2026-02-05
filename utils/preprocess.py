import pandas as pd


def clean_dataframe(df: pd.DataFrame):

    df = df.copy()

    # drop empty columns
    df = df.dropna(axis=1, how="all")

    # convert dates → string (GAN safe)
    for col in df.columns:
        if "date" in col.lower():
            df[col] = df[col].astype(str)

    # fill nulls safely
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].fillna(df[col].median())

    # drop ID-like columns (too unique)
    for col in df.columns:
        if "id" in col.lower():
            df = df.drop(columns=[col])

    return df
