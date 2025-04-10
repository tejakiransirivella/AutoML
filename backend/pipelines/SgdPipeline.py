from ConfigSpace import AndConjunction, Categorical, Configuration, ConfigurationSpace, EqualsCondition, Float, Integer
from sklearn.linear_model import SGDClassifier
from ConfigSpace.conditions import InCondition
import backend.pipelines.util as util
from backend.pipelines.BasePipeline import BasePipeline

class SgdPipeline(BasePipeline):

    def __init__(self):
        super().__init__()
        self.name = "Sgd"

    def config_space(self,configspace:ConfigurationSpace):
        alpha = Float(f"{self.name}.alpha", (1e-7, 0.1), default=0.0001, log=True)
        average = Categorical(f"{self.name}.average", [False, True], default=False)
        epsilon = Float(f"{self.name}.epsilon", (1e-5, 0.1), default=0.0001, log=True)
        eta0 = Float(f"{self.name}.eta0", (1e-7, 0.1), default=0.01, log=True)
        l1_ratio = Float(f"{self.name}.l1_ratio", (1e-9, 1.0), default=0.15,log=True)
        learning_rate = Categorical(f"{self.name}.learning_rate", ["constant", "optimal", "invscaling"], default="invscaling")
        loss = Categorical(f"{self.name}.loss", ["hinge", "log_loss", "modified_huber","squared_hinge",
                                        "perceptron"], default="log_loss")
        penalty = Categorical(f"{self.name}.penalty", ["l2", "l1", "elasticnet"], default="l2")
        power_t = Float(f"{self.name}.power_t", (1e-5, 1.0), default=0.5)
        tol = Float(f"{self.name}.tol", (1e-5, 0.1), default=0.0001, log=True)

        hyperparameters = [alpha, average, epsilon, eta0, l1_ratio, learning_rate, loss, penalty, power_t, tol]
        hyperparameters_cond = [l1_ratio, epsilon, power_t, eta0]

        configspace.add(hyperparameters)

        for param in hyperparameters:
            if param not in hyperparameters_cond:
                configspace.add(EqualsCondition(param, configspace.get("algorithm"), self.name))

        configspace.add(AndConjunction(EqualsCondition(l1_ratio, configspace.get("algorithm"), f"{self.name}"),
                                       EqualsCondition(l1_ratio, penalty, "elasticnet")))
        configspace.add(AndConjunction(EqualsCondition(epsilon, configspace.get("algorithm"), f"{self.name}"),
                                       InCondition(epsilon, loss, ["modified_huber"])))
        configspace.add(AndConjunction(EqualsCondition(power_t, configspace.get("algorithm"), f"{self.name}"),
                                       EqualsCondition(power_t, learning_rate, "invscaling")))
        configspace.add(AndConjunction(EqualsCondition(eta0, configspace.get("algorithm"), f"{self.name}"),
                                       InCondition(eta0, learning_rate, ["invscaling", "constant"])))


    def get_model_for_config(self, config: Configuration,budget:int,seed:int=0):
        config = util.get_config_for_model(self.name,config)
        model = SGDClassifier(**config,max_iter=budget, random_state=seed)
        return model
