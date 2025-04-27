from backend.pipelines.BuildConfigurations import BuildConfigurations
from smac import MultiFidelityFacade
from smac import Scenario
from smac.intensifier.successive_halving import SuccessiveHalving
from ConfigSpace import Configuration
from backend.pipelines.PipelineRegistry import PipelineRegistry
import uuid

from smac.initial_design.abstract_initial_design import AbstractInitialDesign
from smac.initial_design.sobol_design import SobolInitialDesign
class Optimizer:

    def __init__(self,X_train, y_train,pipeline_registry:PipelineRegistry,portfolio):
        self.X_train = X_train
        self.y_train = y_train
        self.pipeline_registry = pipeline_registry
        self.portfolio = portfolio
        self.build_configurations = BuildConfigurations(pipeline_registry)
        self.configspace = self.build_configurations.build_configurations()
        self.initial_configs = [Configuration(self.configspace, values=config) for config in self.portfolio]

    class PortfolioInitialDesign(AbstractInitialDesign):

        def __init__(self, scenario, initial_configs, rng):
            super().__init__(
                scenario=scenario,
                n_configs=len(initial_configs),
                seed=rng,
                additional_configs=[],  # no extra random configs
                n_configs_per_hyperparameter=None,
                max_ratio=1.0,  # disable max_ratio limitation
            )
            self.initial_configs = initial_configs
        
        def _select_configurations(self):
            return self.initial_configs

    def train(self,config:Configuration,budget:float, seed:int = 0, instance: str = None) -> float:
        pipeline = self.pipeline_registry.get_pipeline(config["algorithm"])
        return pipeline.train(self.X_train, self.y_train, config, int(budget),seed)
    
    def optimize(self, kwargs) -> Configuration:
        scenario  = Scenario(configspace=self.configspace, **kwargs,
                             deterministic=True,output_directory=f"smac_output/{uuid.uuid4().hex}")
        initial_design = self.PortfolioInitialDesign(scenario=scenario, initial_configs=self.initial_configs, rng=scenario.seed)
        intensifier = SuccessiveHalving(scenario=scenario, eta=4)
        smac = MultiFidelityFacade(scenario=scenario,target_function=self.train,intensifier=intensifier,initial_design=initial_design)
        best_config = smac.optimize()
        val_score = 1.0-smac.runhistory.get_cost(best_config)
        return (best_config,val_score)
        

        

    
        

