from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

## Configurations
train_filename = "./data/cleveland_train_scaled.csv"
targets = ["target"]
param = [
    {
        'n_neighbors': range(1, 21),
        'p': [1, 2, 3, 4, 5]
    },
]

train_dataset = pd.read_csv(train_filename)
x_train = train_dataset.drop(columns=targets)
t_train = train_dataset[targets] # real

gs = GridSearchCV(
    KNeighborsClassifier(),
    param,
    scoring='recall_macro',
    verbose=True
)

gs.fit(x_train, t_train.squeeze())

# Print the best parameters
print("Best parameters found: ", gs.best_params_)
print("Best score: ", gs.best_score_)