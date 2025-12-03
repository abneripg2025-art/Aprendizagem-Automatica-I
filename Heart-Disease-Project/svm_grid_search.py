import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn import svm

## Configurations
train_filename = "./data/cleveland_train_scaled.csv"
targets = ["target"]

# C_values_search = np.logspace(-3, 3, 7) # 0.001, 0.01, 0.1, 1, 10, 100, 1000
#
# param = [
#     {
#         'C': C_values_search,
#         'kernel': ['linear', 'rbf', 'sigmoid']
#     },
#     {
#         'C': C_values_search,
#         'kernel': ['poly'],
#         'degree': range(2, 5)
#     },
#     # ...
# ]

#C_values_search = [0.1, 0.5, 1, 5, 10]
C_values_search = np.arange(0.2, 0.9, 0.1)

param = [
    {
        'C': C_values_search,
        'kernel': ['poly']
    },
    # ...
]


train_dataset = pd.read_csv(train_filename)
x_train = train_dataset.drop(columns=targets)
t_train = train_dataset[targets] # real

gs = GridSearchCV(
    svm.SVC(),
    param,
    scoring='f1_macro',
    verbose=True
)

gs.fit(x_train, t_train.squeeze())

# Print the best parameters
print("Best parameters found: ", gs.best_params_)
print("Best score: ", gs.best_score_)