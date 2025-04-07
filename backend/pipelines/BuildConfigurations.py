from ConfigSpace import Categorical, ConfigurationSpace
from backend.pipelines.PipelineRegistry import PipelineRegistry
from backend.pipelines.ExtraTreesPipeline import ExtraTreesPipeline
from backend.pipelines.HistGradientBoostingPipeline import HistGradientBoostingPipeline
from backend.pipelines.MLPPipeline import MLPPipeline
from backend.pipelines.PassiveAggressivePipeline import PassiveAggressivePipeline
from backend.pipelines.RandomForestPipeline import RandomForestPipeline
from backend.pipelines.SgdPipeline import SgdPipeline


class BuildConfigurations:

    def __init__(self):
        self.configspace:ConfigurationSpace = ConfigurationSpace(name="automl_configspace")
        self.pipeline_registry = PipelineRegistry()

    def register_pipelines(self):
        self.pipeline_registry.register_pipeline("ExtraTrees", ExtraTreesPipeline())
        self.pipeline_registry.register_pipeline("HistGradientBoosting", HistGradientBoostingPipeline())
        self.pipeline_registry.register_pipeline("MLP", MLPPipeline())
        self.pipeline_registry.register_pipeline("PassiveAggressive", PassiveAggressivePipeline())
        self.pipeline_registry.register_pipeline("RandomForest", RandomForestPipeline())
        self.pipeline_registry.register_pipeline("Sgd", SgdPipeline())

    def build_configurations(self):
        # Define the configuration space for the pipeline
        algorithms = ["ExtraTrees", "HistGradientBoosting","MLP","PassiveAggressive",
                      "RandomForest","Sgd"]
        algorithm = Categorical("algorithm", algorithms)
        self.configspace.add(algorithm)

        self.register_pipelines()

        pipelines = self.pipeline_registry.list_pipelines()
        for pipeline_name in pipelines:
            pipeline = self.pipeline_registry.get_pipeline(pipeline_name)
            pipeline.config_space(self.configspace)
        return self.configspace

    def get_config_space(self):
        return self.configspace

       
