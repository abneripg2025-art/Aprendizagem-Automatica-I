import pandas as pd

dataset_filename = "data/cleveland.csv"
dataset = pd.read_csv(dataset_filename)

dataset = dataset.replace("?", pd.NA)

# Converte todas as colunas possíveis para numérico
for col in dataset.columns:
    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = [c for c in dataset.columns if c not in continuous_features + ["target"]]

# Preenche contínuas com a média
dataset[continuous_features] = dataset[continuous_features].fillna(
    dataset[continuous_features].mean()
)

# Preenche categóricas com a moda
for col in categorical_features:
    dataset[col] = dataset[col].fillna(dataset[col].mode()[0])
