from numpy import sort
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

## Configurations
train_filename = "./data/iris_train_scaled.csv"
test_filename = "./data/iris_test_scaled.csv"
targets = ["class"]
n_neighbors = 1

train_dataset = pd.read_csv(train_filename)
test_dataset = pd.read_csv(test_filename)

x_train = train_dataset.drop(columns=targets)
t_train = train_dataset[targets] # real

x_test = test_dataset.drop(columns=targets)
t_test = test_dataset[targets] # real

knn = KNeighborsClassifier(n_neighbors)
knn.fit(x_train, t_train.squeeze())

y_train = knn.predict(x_train) # model output
y_test = knn.predict(x_test) # model output

classes = train_dataset['class'].unique()

def display_confusion_matrix(real, model_output, classes, title):
    cm = ConfusionMatrixDisplay.from_predictions(real, model_output, labels=sort(classes))
    cm.ax_.set_title(title)
    plt.show()

def display_performance_metrics (real, model_output, classes, title):
    display_confusion_matrix(real, model_output, classes, title)
    accuracy = accuracy_score(real, model_output)
    print (f"{title} accuracy: {accuracy * 100:.2f}%")

display_confusion_matrix(t_train, y_train, classes, "Train confusion matrix")
display_confusion_matrix(t_test, y_test, classes, "Test confusion matrix")






