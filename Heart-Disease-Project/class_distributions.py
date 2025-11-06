import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils import num_groups, show_variable_plots

## Configurations
dataset_filename = "data/cleveland.csv"
num_outputs = 1
variables = None # Histogram variables to show. None to show all
max_rows = 2
max_cols = 2

dataset = pd.read_csv(dataset_filename)

columns = dataset.columns

def plot_distributions(variables, rows, columns):
    fig, axes = plt.subplots(rows, columns)
    axes = axes.flatten() # Flatten in case of a single row or column

    for i, col in enumerate(variables):
        sns.kdeplot(
            data=dataset,
            x=col,
            hue='target',
            multiple="stack",
            ax = axes[i]
        )

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    plt.show()

num_inputs = len(columns) - num_outputs
input_columns = columns[0:num_inputs]

if variables is None:
    num_inputs = len(columns) - num_outputs
    show_variable_plots(columns[0:num_inputs], plot_distributions, max_rows, max_cols)
else:
    show_variable_plots(variables, plot_distributions, max_rows, max_cols)
