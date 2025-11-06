import matplotlib.pyplot as plt
import pandas as pd
from utils import show_variable_plots

dataset_filename = "data/cleveland.csv"
num_outputs = 1
variables = None
max_rows = 2
max_cols = 2
bins = 30

dataset = pd.read_csv(dataset_filename)

for col in dataset.columns:
    dataset[col] = pd.to_numeric(dataset[col], errors="coerce")

columns = dataset.columns

def plot_histograms(variables, rows, columns):
    fig, axes = plt.subplots(rows, columns)
    axes = axes.flatten()

    for i, col in enumerate(variables):
        if dataset[col].dropna().empty:
            continue
        axes[i].hist(dataset[col].dropna(), bins=bins, edgecolor='k', alpha=0.7)
        axes[i].set_title(f"{col} histogram")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        axes[i].grid(False)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

num_inputs = len(columns) - num_outputs
input_columns = columns[0:num_inputs]

if variables is None:
    show_variable_plots(input_columns, plot_histograms, max_rows, max_cols)
else:
    show_variable_plots(variables, plot_histograms, max_rows, max_cols)
