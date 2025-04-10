from ConfigSpace import Categorical, Configuration, ConfigurationSpace, EqualsCondition, Float, Integer
from sklearn.neural_network import MLPClassifier
from backend.pipelines.BasePipeline import BasePipeline
import backend.pipelines.util as util

class MLPPipeline(BasePipeline):

    def __init__(self):
        super().__init__()
        self.name = "MLP"
   
    def config_space(self,configspace:ConfigurationSpace):
        activation = Categorical(f"{self.name}.activation", ["tanh", "relu"], default="relu")
        alpha = Float(f"{self.name}.alpha", (1e-7, 0.1), default=0.0001, log=True)
        early_stopping = Categorical(f"{self.name}.early_stopping", ["valid","train"], default="valid")
        hidden_layer_depth = Integer(f"{self.name}.hidden_layer_depth", (1, 3), default=1)
        learning_rate_init = Float(f"{self.name}.learning_rate_init", (0.0001, 0.5), default=0.001, log=True)
        num_nodes_per_layer = Integer(f"{self.name}.num_nodes_per_layer", (16, 264), default=32, log=True)

        hyperparameters = [activation, alpha, early_stopping, hidden_layer_depth, learning_rate_init, num_nodes_per_layer]

        configspace.add(hyperparameters)

        for param in hyperparameters:
            configspace.add(EqualsCondition(param, configspace.get("algorithm"), self.name)) 

    def get_model_for_config(self, config: Configuration,budget:int,seed:int=0):

        config = util.get_config_for_model(self.name,config)
        if config["early_stopping"] == "valid":
            config["early_stopping"] = True
            config["validation_fraction"] = 0.1
            config["n_iter_no_change"] = 10
        elif config["early_stopping"] == "train":
            config["early_stopping"] = False

        config["hidden_layer_sizes"] = tuple([config["num_nodes_per_layer"]] * config["hidden_layer_depth"])
        del config["num_nodes_per_layer"]
        del config["hidden_layer_depth"]

        model = MLPClassifier(**config,max_iter=budget,random_state=seed)

        return model
     

