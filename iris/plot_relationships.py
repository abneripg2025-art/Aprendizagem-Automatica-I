import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

## Configurations
dataset_filename = "./data/iris.csv"

dataset = pd.read_csv(dataset_filename)

sns.jointplot(
    data=dataset,
    x='sepal_length',
    y='sepal_width',
    hue='class'
)
plt.show()

sns.jointplot(
    data=dataset,
    x='petal_length',
    y='petal_width',
    hue='class'
)
plt.show()