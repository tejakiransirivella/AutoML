from ConfigSpace import Configuration
from sklearn.model_selection import cross_val_score
import numpy as np

class BasePipeline:

    def train(self, X_train, y_train, config: Configuration, budget: int, seed:int=0) -> float:
        """
        Train the model and return the validation score.
        """
        try:
            model = self.get_model_for_config(config,budget,seed)
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

        