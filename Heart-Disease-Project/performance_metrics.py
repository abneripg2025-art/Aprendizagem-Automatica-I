from numpy import sort
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

def display_confusion_matrix(real, model_output, classes, title):
    cm = ConfusionMatrixDisplay.from_predictions(
        real,
        model_output,
        labels=sort(classes)
    )
    cm.ax_.set_title(title)
    plt.show()


def display_performance_metrics(real, model_output,classes, title):
    display_confusion_matrix(real, model_output, classes, title)

    report = classification_report(
        real,
        model_output,
        digits=4
    )
    print(report)
