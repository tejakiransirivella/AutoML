from ConfigSpace import AndConjunction, Categorical, Configuration, ConfigurationSpace, EqualsCondition, Float, Integer
from sklearn.ensemble import HistGradientBoostingClassifier
from ConfigSpace.conditions import InCondition
from backend.pipelines.BasePipeline import BasePipeline
import backend.pipelines.util as util


class HistGradientBoostingPipeline(BasePipeline):

    def __init__(self):
        super().__init__()
        self.name = "HistGradientBoosting"

    def config_space(self,configspace:ConfigurationSpace):
        
        early_stopping = Categorical(f"{self.name}.early_stopping", ["off","valid","train"], default="off")
        l2_regularization = Float(f"{self.name}.l2_regularization", (1e-10, 1.0), default=1e-4, log=True)
        learning_rate = Float(f"{self.name}.learning_rate", (0.01, 1.0), default=0.1, log=True)
        max_leaf_nodes = Integer(f"{self.name}.max_leaf_nodes", (3, 2047), default=31, log=True)
        min_samples_leaf = Integer(f"{self.name}.min_samples_leaf", (1, 200), default=20, log=True)
        n_iter_no_change = Integer(f"{self.name}.n_iter_no_change", (1, 20), default=10)
        validation_fraction = Float(f"{self.name}.validation_fraction", (0.01, 0.4), default=0.1)

        hyperparameters = [early_stopping, l2_regularization, learning_rate, max_leaf_nodes, min_samples_leaf, n_iter_no_change, validation_fraction]
        hyperparameters_cond = [n_iter_no_change, validation_fraction]

        configspace.add(hyperparameters)


        for param in hyperparameters:
            if param not in hyperparameters_cond:
                configspace.add(EqualsCondition(param, configspace.get("algorithm"), self.name)) 

        configspace.add(AndConjunction(EqualsCondition(n_iter_no_change, configspace.get("algorithm"), self.name),
                                       InCondition(n_iter_no_change,early_stopping,["valid","train"])))
        configspace.add(AndConjunction(EqualsCondition(validation_fraction, configspace.get("algorithm"),  self.name),
                                       InCondition(validation_fraction,early_stopping,["valid"])))

       

    def get_model_for_config(self, config: Configuration,budget:int,seed:int=0):
    
        config = util.get_config_for_model(self.name,config)
        if config["early_stopping"] == "valid":
            config["early_stopping"] = True
            config["validation_fraction"] = 0.1
            config["n_iter_no_change"] = 10
        elif config["early_stopping"] == "train" or config["early_stopping"] == "off":
            config["early_stopping"] = False
        model = HistGradientBoostingClassifier(**config,max_iter=budget,random_state=seed)
        return model