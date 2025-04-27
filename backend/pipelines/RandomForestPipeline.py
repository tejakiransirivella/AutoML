from ConfigSpace import Categorical, Configuration, ConfigurationSpace, EqualsCondition, Float, Integer
from sklearn.ensemble import RandomForestClassifier
import backend.pipelines.util as util
from backend.pipelines.BasePipeline import BasePipeline


class RandomForestPipeline(BasePipeline):

     def __init__(self):
        super().__init__()
        self.name = "RandomForest"

     def config_space(self,configspace:ConfigurationSpace):
          bootstrap = Categorical(f"{self.name}.bootstrap",[True, False], default=True)
          criterion = Categorical(f"{self.name}.criterion",["gini", "entropy"], default="gini")
          max_features = Float(f"{self.name}.max_features", (0.0, 1.0), default=0.5)
          min_samples_leaf = Integer(f"{self.name}.min_samples_leaf", (1, 20), default=1)
          min_samples_split = Integer(f"{self.name}.min_samples_split", (2, 20), default=2)

          hyperparameters = [bootstrap, criterion, max_features, min_samples_leaf, min_samples_split]
          configspace.add(hyperparameters)

          for param in hyperparameters:
               configspace.add(EqualsCondition(param, configspace.get("algorithm"), self.name)) 

     def get_model_for_config(self, config: Configuration,budget:int=None,seed:int=0):
          config = util.get_config_for_model(self.name,config)
          model = RandomForestClassifier(**config,**({"n_estimators": budget} if budget is not None else {}), random_state=seed)
          return model 