import pandas as pd

## Configurations
dataset_filename = "data/cleveland_train_scaled.csv"

dataset = pd.read_csv(dataset_filename)

# Check data types and missing data
dataset.info()

# Print statistic information
print(dataset.describe(include="all"))