import openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from backend.Optimizer import Optimizer
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
import backend.meta_learning.preprocess as preprocess
import smac
import numpy as np


dataset = openml.datasets.get_dataset(31)  # "credit-g"
# X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
# print(X.head())
df = preprocess.load_dataset(31)
# print(df.head())
y = df["class"]
y = y.astype('category').cat.codes
X = df.drop(columns=["class"])
X = pd.get_dummies(X, drop_first=False).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scores = cross_val_score(SGDClassifier(), X_train, y_train, cv=5)
# print(scores)
optimizer = Optimizer(X_train,y_train,seed=42)
best_config = optimizer.optimize()
print(best_config)

# loss = runhistory.get_cost(best_config)
# print(loss)

# model = RandomForestClassifier(bootstrap=True,criterion="entropy",max_features=0.261397316560,min_samples_leaf=6,
#                        min_samples_split=6,n_estimators=100).fit(X_train,y_train)

# scores = cross_val_score(model, X_test, y_test, cv=5)
# score = 1-np.mean(scores)
# print(1-score)