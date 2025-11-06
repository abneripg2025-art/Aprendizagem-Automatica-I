import pandas as pd

## Configurations
dataset_input_filename = "./data/bezdekIris.data"
dataset_output_filename = "./data/iris.csv"
column_names = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class",
]

dataset = pd.read_csv(
    dataset_input_filename,
    header=None,
    names = column_names
)

dataset.to_csv(
    dataset_output_filename,
    index=False,
)