import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

dataset_input_filename = "./data/cleveland.data"
dataset_output_filename = "./data/cleveland_preprocessed.csv"

column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

dataset = pd.read_csv(
    dataset_input_filename,
    header=None,
    names=column_names,
    na_values="?"
)

dataset["ca"] = pd.to_numeric(dataset["ca"])
dataset["thal"] = pd.to_numeric(dataset["thal"])

dataset = dataset.dropna().reset_index(drop=True)

X = dataset.drop(columns=["target"])
y = dataset["target"]

scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "./models/cleveland_scaler.save")

dataset_preprocessed = pd.DataFrame(X_scaled, columns=X.columns)
dataset_preprocessed["target"] = y.values

dataset_preprocessed.to_csv(dataset_output_filename, index=False)
