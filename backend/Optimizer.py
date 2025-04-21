from backend.pipelines.BuildConfigurations import BuildConfigurations
from smac import MultiFidelityFacade
from smac import Scenario
from ConfigSpace import Configuration
from backend.pipelines.PipelineRegistry import PipelineRegistry
import uuid

class Optimizer:

    def __init__(self,X_train, y_train,pipeline_registry:PipelineRegistry):
        self.X_train = X_train
        self.y_train = y_train
        self.pipeline_registry = pipeline_registry
        self.build_configurations = BuildConfigurations(pipeline_registry)
        self.configspace = self.build_configurations.build_configurations()


    def train(self,config:Configuration,budget:float, seed:int = 0, instance: str = None) -> float:
        pipeline = self.pipeline_registry.get_pipeline(config["algorithm"])
        return pipeline.train(self.X_train, self.y_train, config, int(budget),seed)
    
    def optimize(self, kwargs) -> Configuration:
        scenario  = Scenario(configspace=self.configspace, **kwargs,
                             deterministic=True,output_directory=f"smac_output/{uuid.uuid4().hex}")
        smac = MultiFidelityFacade(scenario=scenario,target_function=self.train)
        best_config = smac.optimize()
        val_score = 1.0-smac.runhistory.get_cost(best_config)
        return (best_config,val_score)
        

        

    
        

