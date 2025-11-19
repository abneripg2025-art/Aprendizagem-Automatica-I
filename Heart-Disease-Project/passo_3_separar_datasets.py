import pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

## Configurations
dataset_filename = "data/cleveland.csv"
train_dataset = "data/cleveland_train_scaled.csv"
test_dataset = "data/cleveland_test_scaled.csv"
targets = ["target"]
perc_train = 50/100

dataset = pd.read_csv(dataset_filename)

x = dataset.drop(columns=targets)
t = dataset[targets]

column_names = x.columns

continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

categorical_features = [col for col in x.columns if col not in continuous_features]

x_train, x_test, t_train, t_test = train_test_split(
    x, t,
    train_size = perc_train,
    stratify = t
)

scaler = MinMaxScaler((-1, 1)).fit(x_train[continuous_features])
# scaler = StandardScaler().fit(x_train[continuous_features])

x_train_scaled_cont = pd.DataFrame(
    scaler.transform(x_train[continuous_features]),
    columns=continuous_features,
    index=x_train.index
)

x_test_scaled_cont = pd.DataFrame(
    scaler.transform(x_test[continuous_features]),
    columns=continuous_features,
    index=x_test.index
)

x_train_cat = x_train[categorical_features]
x_test_cat = x_test[categorical_features]

x_train_scaled = pd.concat([x_train_scaled_cont, x_train_cat], axis='columns')
x_test_scaled = pd.concat([x_test_scaled_cont, x_test_cat], axis='columns')

train = pd.concat([x_train_scaled, t_train], axis='columns')
test = pd.concat([x_test_scaled, t_test], axis='columns')

train.to_csv(train_dataset, index=False)
test.to_csv(test_dataset, index=False)
