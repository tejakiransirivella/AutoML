from ConfigSpace import Categorical, Configuration, ConfigurationSpace, EqualsCondition, Float, Integer
from sklearn.linear_model import PassiveAggressiveClassifier
import backend.pipelines.util as util
from backend.pipelines.BasePipeline import BasePipeline

class PassiveAggressivePipeline(BasePipeline):

    def __init__(self):
        super().__init__()
        self.name = "PassiveAggressive"
   
    def config_space(self,configspace:ConfigurationSpace):
        C = Float(f"{self.name}.C", (1e-5, 10.0), default=1.0, log=True)
        average = Categorical(f"{self.name}.average", [False, True], default=False)
        loss = Categorical(f"{self.name}.loss", ["hinge", "squared_hinge"], default="hinge")
        tol = Float(f"{self.name}.tol", (1e-5, 0.1), default=0.0001, log=True)

        hyperparameters = [C, average, loss, tol]
        configspace.add(hyperparameters)

        for param in hyperparameters:
            configspace.add(EqualsCondition(param, configspace.get("algorithm"), self.name)) 
    
    def get_model_for_config(self, config: Configuration,budget:int, seed:int=0):
        config = util.get_config_for_model(f"{self.name}",config)
        model = PassiveAggressiveClassifier(**config,max_iter=budget, random_state=seed)
        return model



