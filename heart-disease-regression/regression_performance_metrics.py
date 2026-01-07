import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    max_error,
    accuracy_score,
    confusion_matrix
)
import matplotlib.pyplot as plt

def plot_targets_versus_model(t, y, title):
        t = np.asarray(t).ravel()
        y = np.asarray(y).ravel()

        classes = np.unique(t)
        means = [y[t == c].mean() for c in classes]

        plt.plot(classes, means, marker="o")
        plt.plot([0, 4], [0, 4], linestyle="--")
        plt.xlabel("Real disease level")
        plt.ylabel("Mean predicted disease level")
        plt.title(title)
        plt.grid(True)
        plt.show()



def display_performance_metrics(t, y, title):
    print(title)

    mae = mean_absolute_error(t, y)
    mse = mean_squared_error(t, y)
    rmse = np.sqrt(mse)
    maxerror = max_error(t, y)

    print(f"Mean Absolute Error (MAE) .... : {mae:.4f}")
    print(f"Mean Squared Error (MSE) ..... : {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE) : {rmse:.4f}")
    print(f"Max error .................... : {maxerror:.4f}")

    y_class = np.clip(np.round(y), 0, 4).astype(int)

    acc = accuracy_score(t, y_class)
    cm = confusion_matrix(t, y_class)

    print(f"Classification Accuracy ..... : {acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    print()

    plot_targets_versus_model(t, y, title)
