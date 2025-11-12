import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

train_filename = "./data/iris_train_scaled.csv"
test_filename = "./data/iris_test_scaled.csv"
targets = ["class"]

train_dataset = pd.read_csv(train_filename)
test_dataset = pd.read_csv(test_filename)

x_train = train_dataset.drop(columns=targets)
t_train = train_dataset[targets] # real

params = [ {'n_neighbors': range(1,21),
     'p': [1,2,3,4,5]}]

gs = GridSearchCV(KNeighborsClassifier(), params,
                  scoring='f1_macro', verbose=True)

gs.fit(x_train, t_train.squeeze())

print("Best parameters found: ", gs.best_params_)
print("Best score: ", gs.best_score_)