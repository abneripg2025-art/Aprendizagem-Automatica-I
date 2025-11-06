import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import show_variable_plots

## Configurations
dataset_filename = "./data/iris.csv"
num_outputs = 1
variables = None # Histogram variables to show. None to show all
max_rows = 2
max_cols = 2

dataset = pd.read_csv(dataset_filename)

columns = dataset.columns

def plot_outliers(variables, rows, columns):
    fig, axes = plt.subplots(rows, columns)
    axes = axes.flatten() # Flatten in case of a single row or column

    for i, col in enumerate(variables):
        sns.boxplot(
            x=dataset[col],
            ax=axes[i]
        )
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_xlabel(col)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    plt.show()

num_inputs = len(columns) - num_outputs
input_columns = columns[0:num_inputs]

if variables is None:
    num_inputs = len(columns) - num_outputs
    show_variable_plots(columns[0:num_inputs], plot_outliers, max_rows, max_cols)
else:
    show_variable_plots(variables, plot_outliers, max_rows, max_cols)
