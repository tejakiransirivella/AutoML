import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from backend.Optimizer import Optimizer
from backend.meta_learning import preprocess
from backend.pipelines.PipelineRegistry import PipelineRegistry


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
       

    def one_hot_encoding(self,X_train, y_train):
        
        y_train = y_train.astype('category').cat.codes
        X_train = pd.get_dummies(X_train, drop_first=False).astype(int)
        return X_train, y_train

    def fit(self,X_train,y_train):
       
        self.X_train, self.y_train = self.one_hot_encoding(X_train, y_train)
        self.optimizer = Optimizer(self.X_train,self.y_train,self.pipeline_registry)
        self.best_config = self.optimizer.optimize(self.kwargs)
        print(self.best_config)
    
    def predict(self,X_test):
        if self.best_config is None:
            raise ValueError("Model has not been trained yet. Call fit() before predict().")
        
        X_test = pd.get_dummies(X_test, drop_first=False).astype(int)
        
        pipeline = self.pipeline_registry.get_pipeline(self.best_config["algorithm"])
        y_pred = pipeline.predict(self.X_train,self.y_train,X_test, self.best_config, self.kwargs["max_budget"],self.kwargs["seed"])
        return y_pred
        
    
def main():
    df = preprocess.load_dataset(31)
    y = df["class"]
    X = df.drop(columns=["class"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    autoclassifier = AutoClassifier(seed=42,walltime_limit=60,min_budget = 100, max_budget = 1000)
    autoclassifier.fit(X_train,y_train)
    X_test,y_test = autoclassifier.one_hot_encoding(X_test,y_test)
    y_pred = autoclassifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

main()

          