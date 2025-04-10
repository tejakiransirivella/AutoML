import autosklearn.classification
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Fit Auto-sklearn
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=300,  # Total time in seconds
    per_run_time_limit=60,        # Time per model
    memory_limit=4096,            # RAM per job (MB)
    seed=42,
)

automl.fit(X_train, y_train)

# Predict and evaluate
y_pred = automl.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print best configuration
print("\nBest Configuration:")
print(automl.show_models())

# Loss value (e.g., 1 - accuracy)
print("\nLoss of best model:", 1 - accuracy)
