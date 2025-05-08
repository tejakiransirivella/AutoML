from autosklearn.pipeline.components.classification import _classifiers
import openml
import pandas as pd

from backend.config import Config
from backend.meta_learning.preprocess import Preprocess
# print(_classifiers.keys())

import autosklearn.classification
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
config = Config()
preprocess = Preprocess(config.get_test_path())
df = preprocess.load_dataset("1067")
target = openml.datasets.get_dataset(1067).default_target_attribute
y = df[target]
X = df.drop(columns=[target])

# print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# Fit Auto-sklearn
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=600,  # Total time in seconds
    seed=42,
    memory_limit=6144
)


automl.fit(X_train, y_train)

results = pd.DataFrame(automl.cv_results_)
best_index = results['rank_test_scores'].argmin()  # rank 1 is best
best_score = results.loc[best_index, 'mean_test_score']
best_config = results.loc[best_index]

print(best_index,best_score)

# Predict and evaluate
y_pred = automl.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# # Print best configuration
# print("\nBest Configuration:")
# print(best_config)
# # print(automl.show_models())

# Loss value (e.g., 1 - accuracy)
# print("\nLoss of best model:", 1 - accuracy)
