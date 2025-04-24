from collections import Counter
from ConfigSpace import Configuration
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np

class BasePipeline:

    def train(self, X_train, y_train, config: Configuration, budget: int, seed:int=0) -> float:
        """
        Train the model and return the validation score.
        """
        try:
            model = self.get_model_for_config(config,budget,seed)
            label_counts = Counter(y_train)
            min_class_count = min(label_counts.values())    
            if min_class_count < 5:
                print(f"Class imbalance detected. Minimum class count: {min_class_count}")
                print("Using 80/20 holdout split for training and validation.")
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
                model.fit(X_train, y_train)
                val_score = model.score(X_val, y_val)
                return 1-val_score
            else:
                scores = cross_val_score(model, X_train, y_train, cv=5)
                return 1-np.mean(scores)
        except Exception as e:
            print(f"Training failed for config {config}")
            print("Error: ", e)
            return float("inf")
    
    def predict(self, X_train,y_train,X_test, config: Configuration, budget: int, seed:int=0) -> float:
        """
        test the model and return the validation score.
        """
        model = self.get_model_for_config(config,budget,seed)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return predictions

        