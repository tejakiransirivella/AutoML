import json
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import openml

from backend.Optimizer import Optimizer
from backend.pipelines.PipelineRegistry import PipelineRegistry
from backend.config import Config
from backend.meta_learning.preprocess import Preprocess


class AutoClassifier:

    def __init__(self,**kwargs):
        """
        Keyword Args:
            - walltime_limit (int): Optional wall time limit in seconds
            - min_budget (int): Optional minimum budget
            - max_budget (int): Optional maximum budget
            - seed (int): Optional random seed
            - deterministic (bool): Whether to run deterministically
            """
        self.kwargs = kwargs
        self.best_config = None
        self.pipeline_registry = PipelineRegistry()
        self.X_train = None
        self.y_train = None
        self.val_score = None
       
    def load_portfolio(self):
        config = Config()
        with open(f"{config.get_results_path()}/portfolio.json", "r") as f:
            portfolio = json.load(f)["portfolio"]
        return portfolio
        
    def one_hot_encoding(self,X_train, y_train):
        
        y_train = y_train.astype('category').cat.codes
        X_train = pd.get_dummies(X_train, drop_first=False).astype(int)
        X_train = MinMaxScaler().fit_transform(X_train)

        return X_train, y_train

    def fit(self,X_train,y_train):
        portfolio = self.load_portfolio()
        print("Portfolio loaded",len(portfolio))
        self.X_train,self.y_train = X_train, y_train
        self.optimizer = Optimizer(self.X_train,self.y_train,self.pipeline_registry,portfolio)
        results = self.optimizer.optimize(self.kwargs)
        self.best_config = results[0]
        self.val_score = results[1]
    
    def predict(self,X_test):
        if self.best_config is None:
            raise ValueError("Model has not been trained yet. Call fit() before predict().")
        
        pipeline = self.pipeline_registry.get_pipeline(self.best_config["algorithm"])
        y_pred = pipeline.predict(self.X_train,self.y_train,X_test, self.best_config, self.kwargs["max_budget"],self.kwargs["seed"])
        return y_pred
        
    
def main():
    config = Config()
    preprocess = Preprocess(config.get_test_path())
    df = preprocess.load_dataset("1067")
    target = openml.datasets.get_dataset(1067).default_target_attribute
    y = df[target]
    X = df.drop(columns=[target])
    # print(y.columns)
    autoclassifier = AutoClassifier(seed=42,walltime_limit=60,min_budget = 10, max_budget = 200)
    X,y = autoclassifier.one_hot_encoding(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    autoclassifier.fit(X_train,y_train)
    print(autoclassifier.best_config)
    print(autoclassifier.val_score)
    print("Starting prediction")
    y_pred = autoclassifier.predict(X_test)
    accuracy = accuracy_score(y_test ,y_pred)
    print(accuracy)
    # print(dict(autoclassifier.best_config))


if __name__ == "__main__":
    main()

          