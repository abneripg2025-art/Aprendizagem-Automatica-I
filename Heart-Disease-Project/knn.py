from numpy import sort
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

## Configurations
train_filename = "data/cleveland_train_scaled.csv"
test_filename = "data/cleveland_test_scaled.csv"
targets = ["target"]
n_neighbors = 1

train_dataset = pd.read_csv(train_filename)
test_dataset = pd.read_csv(test_filename)

x_train = train_dataset.drop(columns=targets)
t_train = train_dataset[targets] # real

x_test = test_dataset.drop(columns=targets)
t_test = test_dataset[targets] # real

knn = KNeighborsClassifier(n_neighbors)
knn.fit(x_train, t_train)

y_train = knn.predict(x_train) # model output
y_test = knn.predict(x_test) # model output

y_train_bin = (y_train != 0).astype(int)
y_test_bin = (y_test != 0).astype(int)

t_train_bin = (t_train != 0).astype(int)
t_test_bin = (t_test != 0).astype(int)


classes = train_dataset['target'].unique()

def display_confusion_matrix(real, model_output, classes, title):
    cm = ConfusionMatrixDisplay.from_predictions(real, model_output, labels=sort(classes))
    cm.ax_.set_title(title)
    plt.show()

display_confusion_matrix(t_train_bin, y_train_bin, [0, 1], "Train confusion matrix (binary)")
display_confusion_matrix(t_test_bin, y_test_bin, [0, 1], "Test confusion matrix (binary)")
