import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

## Configurations
train_filename = "./data/cleveland_train.csv"
targets = ["target"]

param = [
    {
        'n_estimators': [190, 200, 210],
        'max_samples': [80, 90, 100]
    },
    # ...
]

train_dataset = pd.read_csv(train_filename)
x_train = train_dataset.drop(columns=targets)
t_train = train_dataset[targets] # real

gs = GridSearchCV(
    RandomForestRegressor(),
    param,
    scoring='neg_root_mean_squared_error',
    verbose=True
)

gs.fit(x_train, t_train.squeeze())

# Print the best parameters
print("Best parameters found: ", gs.best_params_)
print("Best score: ", gs.best_score_)
