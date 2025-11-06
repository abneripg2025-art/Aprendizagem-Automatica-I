import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

## Configurations
dataset_filename = "data/cleveland.csv"

dataset = pd.read_csv(dataset_filename)

sns.jointplot(
    data=dataset,
    x='age',
    y='thalach',
    hue='target'
)
plt.show()

sns.jointplot(
    data=dataset,
    x='age',
    y='ca',
    hue='target'
)
plt.show()