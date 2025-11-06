import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

## Configurations
dataset_filename = "data/cleveland.csv"

dataset = pd.read_csv(dataset_filename)

sns.pairplot(
    data=dataset,
    hue='target'
)
plt.show()
