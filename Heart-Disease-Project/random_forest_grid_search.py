import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

## Configurations
train_filename = "./data/cleveland_train_scaled.csv"
targets = ["target"]

# param = [
#     {
#         'n_estimators': [10, 50, 100, 200, 500, 1000],
#         'max_depth': [None, 1, 2, 3, 4],
#         'max_samples': [None, 5, 10, 15, 20, 30, 40],
#     },
# ]

# param = [
#     {
#         'n_estimators': [120, 140, 150, 200, 300, 400],
#         'max_samples': [17, 19, 20, 22, 25],
#     },
# ]

param = [
    {
        'n_estimators': [100, 125, 150, 175, 200],
        'max_samples': [10, 15, 20, 25, 30],
    },
]

train_dataset = pd.read_csv(train_filename)
x_train = train_dataset.drop(columns=targets)
t_train = train_dataset[targets] # real

gs = GridSearchCV(
    RandomForestClassifier(),
    param,
    scoring='f1_macro',
)

gs.fit(x_train, t_train.squeeze())

# Print the best parameters
print("Best parameters found: ", gs.best_params_)
print("Best score: ", gs.best_score_)