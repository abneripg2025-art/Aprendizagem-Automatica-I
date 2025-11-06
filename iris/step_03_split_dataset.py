import pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

## Configurations
dataset_filename = "./data/iris.csv"
train_dataset = "./data/iris_train_scaled.csv"
test_dataset = "./data/iris_test_scaled.csv"
targets = ["class"]
perc_train = 50/100

dataset = pd.read_csv(dataset_filename)

x = dataset.drop(columns=targets)
t = dataset[targets]

column_names = x.columns

x_train, x_test, t_train, t_test = train_test_split(
    x, t,
    train_size = perc_train,
    stratify = t
)

# Data rescale
#scaler = StandardScaler().fit(x_train)
scaler = MinMaxScaler((-1, 1)).fit(x_train)

x_train_scaled = pd.DataFrame(
    scaler.transform(x_train),
    columns= column_names,
    index = x_train.index
)

x_test_scaled = pd.DataFrame(
    scaler.transform(x_test),
    columns= column_names,
    index = x_test.index
)

train = pd.concat([x_train_scaled, t_train], axis='columns', join='inner')
test = pd.concat([x_test_scaled, t_test], axis='columns', join='inner')

train.to_csv(train_dataset, index=False)
test.to_csv(test_dataset, index=False)
