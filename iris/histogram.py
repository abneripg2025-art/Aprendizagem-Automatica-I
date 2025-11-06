import matplotlib.pyplot as plt
import pandas as pd

from utils import show_variable_plots

## Configurations
dataset_filename = "./data/iris_train_scaled.csv"
num_outputs = 1
variables = None # Histogram variables to show. None to show all
max_rows = 2
max_cols = 2
bins = 30

dataset = pd.read_csv(dataset_filename)

columns = dataset.columns

def plot_histograms(variables, rows, columns):
    fig, axes = plt.subplots(rows, columns)
    axes = axes.flatten() # Flatten in case of a single row or column

    for i, col in enumerate(variables):
        axes[i].hist(dataset[col], bins=bins, edgecolor='k', alpha=0.7)
        axes[i].set_title(f"{col} histogram")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        plt.grid(False)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    plt.show()

num_inputs = len(columns) - num_outputs
input_columns = columns[0:num_inputs]

if variables is None:
    num_inputs = len(columns) - num_outputs
    show_variable_plots(columns[0:num_inputs], plot_histograms, max_rows, max_cols)
else:
    show_variable_plots(variables, plot_histograms, max_rows, max_cols)
