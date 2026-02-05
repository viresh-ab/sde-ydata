import pandas as pd
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.preprocessing.regular.processor import RegularDataProcessor


def generate_ydata(df: pd.DataFrame, rows: int, epochs: int):

    num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

    processor = RegularDataProcessor(
        num_cols=num_cols,
        cat_cols=cat_cols
    )

    data = processor.fit_transform(df)

    model = RegularSynthesizer(
        modelname="ctgan",
        epochs=epochs,
        batch_size=128
    )

    model.fit(data)

    synth = model.sample(rows)

    return processor.inverse_transform(synth)
