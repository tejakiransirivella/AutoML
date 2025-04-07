from ConfigSpace import AndConjunction, Categorical, Configuration, ConfigurationSpace, EqualsCondition, Float, Integer
from sklearn.linear_model import SGDClassifier
from ConfigSpace.conditions import InCondition
import backend.pipelines.util as util
from backend.pipelines.BasePipeline import BasePipeline

class SgdPipeline(BasePipeline):

    def config_space(self,configspace:ConfigurationSpace):
        alpha = Float("sgd.alpha", (1e-7, 0.1), default=0.0001, log=True)
        average = Categorical("sgd.average", [False, True], default=False)
        epsilon = Float("sgd.epsilon", (1e-5, 0.1), default=0.0001, log=True)
        eta0 = Float("sgd.eta0", (1e-7, 0.1), default=0.01, log=True)
        l1_ratio = Float("sgd.l1_ratio", (1e-9, 1.0), default=0.15,log=True)
        learning_rate = Categorical("sgd.learning_rate", ["constant", "optimal", "invscaling"], default="invscaling")
        loss = Categorical("sgd.loss", ["hinge", "log_loss", "modified_huber","squared_hinge",
                                        "perceptron"], default="log_loss")
        penalty = Categorical("sgd.penalty", ["l2", "l1", "elasticnet"], default="l2")
        power_t = Float("sgd.power_t", (1e-5, 1.0), default=0.5)
        tol = Float("sgd.tol", (1e-5, 0.1), default=0.0001, log=True)

        hyperparameters = [alpha, average, epsilon, eta0, l1_ratio, learning_rate, loss, penalty, power_t, tol]
        hyperparameters_cond = [l1_ratio, epsilon, power_t, eta0]

        configspace.add(hyperparameters)

        for param in hyperparameters:
            if param not in hyperparameters_cond:
                configspace.add(EqualsCondition(param, configspace.get("algorithm"), "Sgd"))

        configspace.add(AndConjunction(EqualsCondition(l1_ratio, configspace.get("algorithm"), "Sgd"),
                                       EqualsCondition(l1_ratio, penalty, "elasticnet")))
        configspace.add(AndConjunction(EqualsCondition(epsilon, configspace.get("algorithm"), "Sgd"),
                                       InCondition(epsilon, loss, ["modified_huber"])))
        configspace.add(AndConjunction(EqualsCondition(power_t, configspace.get("algorithm"), "Sgd"),
                                       EqualsCondition(power_t, learning_rate, "invscaling")))
        configspace.add(AndConjunction(EqualsCondition(eta0, configspace.get("algorithm"), "Sgd"),
                                       InCondition(eta0, learning_rate, ["invscaling", "constant"])))


    def get_model_for_config(self, config: Configuration,budget:int,seed:int=0):
        config = util.get_config_for_model("sgd",config)
        model = SGDClassifier(**config,max_iter=budget, random_state=seed)
        return model
