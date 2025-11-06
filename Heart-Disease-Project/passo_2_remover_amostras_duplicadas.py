import pandas as pd

## Configurations
dataset_filename = "data/cleveland.csv"

dataset = pd.read_csv(dataset_filename)

initial_rows = len(dataset)
dataset = dataset.drop_duplicates()
rows = len(dataset)
duplicated_rows = initial_rows - rows

if duplicated_rows > 0:
    print(f"Removed {duplicated_rows} duplicated samples")
    dataset.to_csv(
        dataset_filename,
        index=False,
    )

print(f"Dataset contains {rows} samples")
