import pandas as pd
from sklearn import svm

from performance_metrics import display_performance_metrics

## Configurations
train_filename = "data/cleveland_train_scaled.csv"
test_filename = "data/cleveland_test_scaled.csv"
targets = ["target"]
C = 1
kernel = 'rbf'

train_dataset = pd.read_csv(train_filename)
test_dataset = pd.read_csv(test_filename)

x_train = train_dataset.drop(columns=targets)
t_train = train_dataset[targets] # real

x_test = test_dataset.drop(columns=targets)
t_test = test_dataset[targets] # real

model = svm.SVC(
    C = C,
    kernel = kernel
)
model.fit(x_train, t_train.squeeze())

y_train = model.predict(x_train) # model output
y_test = model.predict(x_test) # model output

classes = train_dataset['target'].unique()

display_performance_metrics(t_train, y_train, classes, "Train")
display_performance_metrics(t_test, y_test, classes, "Test")

