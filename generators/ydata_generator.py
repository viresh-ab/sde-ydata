import pandas as pd
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.synthesizers.regular import RegularSynthesizer


def generate_ydata(df: pd.DataFrame, rows: int, epochs: int):

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    model_params = ModelParameters(
        batch_size=100,
        lr=2e-4,
        betas=(0.5, 0.9)
    )

    model = RegularSynthesizer(
        modelname="ctgan",
        model_parameters=model_params
    )

    train_args = TrainParameters(epochs=epochs)

    model.fit(df, train_args, num_cols=num_cols, cat_cols=cat_cols)

    return model.sample(rows)
