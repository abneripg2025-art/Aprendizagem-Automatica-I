import pandas as pd

dataset_input_filename = "data/processed.cleveland.data"
dataset_output_filename = "data/cleveland.csv"

column_names = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

dataset = pd.read_csv(
    dataset_input_filename,
    header=None,
    names=column_names,
    na_values="?",
    encoding = "latin1",
)

dataset["target"] = dataset["target"].apply(lambda x: 1 if x > 0 else 0)

dataset.to_csv(
    dataset_output_filename,
    index=False
)
