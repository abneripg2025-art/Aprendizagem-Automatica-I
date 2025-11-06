import matplotlib.pyplot as plt
import pandas as pd

## Configurations
dataset_filename = "./data/iris_test.csv"
num_outputs = 1

dataset = pd.read_csv(dataset_filename)
columns = dataset.columns
num_inputs = len(columns) - num_outputs
input_columns = columns[0:num_inputs]

class_counts = dataset['class'].value_counts()
rows = len(dataset)

# Plot a pie chart
class_counts.plot.pie(
    startangle=90,
    autopct=lambda percent: f"{percent:.2f}% ({percent / 100 * rows:.0f})"
)
plt.title('Class Proportionality')
plt.ylabel('') # Hide the y-label
plt.show()
