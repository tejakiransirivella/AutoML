import openml
from sklearn.metrics import accuracy_score
from backend.autoclassifier import AutoClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from backend.meta_learning.preprocess import Preprocess
from backend.config import Config

config = Config()
preprocess = Preprocess(config.get_train_path())
df = preprocess.load_dataset("31")
target = openml.datasets.get_dataset(31).default_target_attribute
y = df[target]
X = df.drop(columns=[target])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = AutoClassifier(seed=42, walltime_limit=60, min_budget=10, max_budget=200)
print("Evaluating Framework on dataset:", openml.datasets.get_dataset(31).name)
clf.fit(X_train, y_train)
print("Best configuration:", clf.best_config, "Validation score:", clf.val_score)
print("Starting prediction")
X_test, y_test = clf.one_hot_encoding(X_test, y_test)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test ,y_pred)
print("Test accuracy:", accuracy)
