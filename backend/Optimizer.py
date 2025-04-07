from backend.pipelines.BuildConfigurations import BuildConfigurations
from smac import MultiFidelityFacade
from smac import Scenario
from ConfigSpace import Configuration
from smac.intensifier.successive_halving import SuccessiveHalving

class Optimizer:

    def __init__(self,X_train, y_train,seed:int = 0):
        self.X_train = X_train
        self.y_train = y_train
        self.seed = seed
        self.build_configurations = BuildConfigurations()
        self.configspace = self.build_configurations.build_configurations()

    def train(self,config:Configuration,budget:float, seed:int = 0, instance: str = None) -> float:
        pipeline = self.build_configurations.pipeline_registry.get_pipeline(config["algorithm"])
        return pipeline.train(self.X_train, self.y_train, config, int(budget),seed)
    
    def optimize(self):
        scenario  = Scenario(configspace=self.configspace, walltime_limit=180,min_budget=500,max_budget=2000,seed=self.seed,
                             deterministic=True,output_directory="smac_output")
        smac = MultiFidelityFacade(scenario=scenario,target_function=self.train)
        best_config = smac.optimize()
        return best_config
        

        

    
        

