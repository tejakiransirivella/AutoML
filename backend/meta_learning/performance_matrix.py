import json

import openml
import ray

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ConfigSpace import Categorical, ConfigurationSpace

from backend.config import Config
from backend.autoclassifier import AutoClassifier
from backend.meta_learning.preprocess import Preprocess
from ConfigSpace import Configuration
from backend.pipelines.PipelineRegistry import PipelineRegistry
from backend.pipelines.BuildConfigurations import BuildConfigurations


class ConfigurationCell:

    def __init__(self, config, dataset_id:int=None, row:int=None, col:int=None,val_score:float=None):
        self.config = config
        self.dataset_id = dataset_id
        self.val_score = val_score
        self.row = row
        self.col = col

class PerformanceMatrix:

    def __init__(self, best_candidates_path, data_path:str=None, performance_matrix_path:str=None):
        self.best_candidates_path = best_candidates_path
        self.data_path = data_path
        self.performace_matrix_path = performance_matrix_path
    
    def load_configurations(self):
        configurations = []
        dataset_ids = []
        val_scores = []
        with open(self.best_candidates_path, "r") as f:
            data = json.load(f)
            datasets = data["best_candidates"]
            for dataset in datasets:
                dataset_ids.append(int(dataset["dataset_id"]))
                configurations.append(dataset["best_config"])
                val_scores.append(dataset["val_score"])
                 
        return dataset_ids, configurations, val_scores
    
    def write_performace_matrix(self, performance_matrix):
        with open(self.performace_matrix_path, "w") as f:
            json.dump(
                {"performace_matrix":[
                    performance_matrix[i][j].__dict__
                    for i in range(len(performance_matrix)) 
                    for j in range(len(performance_matrix))
                    if performance_matrix[i][j] is not None
                    ]
                },
                f,indent=4, default=str) 
            
    
    @staticmethod
    @ray.remote
    def process_task(task,config_space:ConfigurationSpace,pipeline_registry:PipelineRegistry,
                     budget:int, seed:int) -> ConfigurationCell:
        configuration_cell = None
        try:
            row,col,config_dict,dataset_id,X,y = task
            print("dataset id : ", dataset_id)
            config = Configuration(config_space, values= config_dict)
            autoclassifier = AutoClassifier()
            X,y = autoclassifier.one_hot_encoding(X,y)
            pipeline = pipeline_registry.get_pipeline(config["algorithm"])
            accuracy = 1.0-pipeline.train(X, y, config,budget,seed)
            configuration_cell = ConfigurationCell(config_dict, dataset_id=dataset_id, row=row, col=col, val_score=accuracy)
            print(f"Configuration: {config} Dataset: {dataset_id} Accuracy: {accuracy}")

        except Exception as e:
            print(f"Error processing config: {task[2]} and dataset: {task[3]}: {e}")
            
        return configuration_cell
    
    def create_performance_matrix(self):
        dataset_ids , configurations , val_scores = self.load_configurations()
        data = []
        preprocess = Preprocess(self.data_path)

        for i in range(len(dataset_ids)):
            dataset_id = dataset_ids[i]
            dataset = preprocess.load_dataset(dataset_id)
            target = openml.datasets.get_dataset(dataset_id).default_target_attribute
            y = dataset[target]
            X = dataset.drop(columns=[target])
            data.append((X, y))
            
        rows = len(configurations)
        performance_matrix = [[None for _ in range(rows)] for _ in range(rows)]
        tasks = []
        for i in range(rows):
            configuration = configurations[i]
            for j in range(rows):
                if i == j:
                    performance_matrix[i][j] = ConfigurationCell(configuration, dataset_ids[j], i, j, val_scores[j])
                else:
                    tasks.append((i,j,configuration,dataset_ids[j],data[j][0],data[j][1]))

        print("No of tasks: ", len(tasks))

        tasks = tasks[:2]

        pipeline_registry:PipelineRegistry = PipelineRegistry()
        build_configurations = BuildConfigurations(pipeline_registry)
        configspace = build_configurations.build_configurations()

        futures = []

        for i in range(len(tasks)):
            future = PerformanceMatrix.process_task.remote(tasks[i],configspace,pipeline_registry,budget = None, seed = 42)
            futures.append(future)

        for future in futures:
            try:
                configuration_cell:ConfigurationCell = ray.get(future)
                if configuration_cell is not None:
                    performance_matrix[configuration_cell.row][configuration_cell.col] = configuration_cell
                    print(f"Configuration cell: {configuration_cell.row} {configuration_cell.col} {configuration_cell.val_score}")
            except ray.exceptions.RayTaskError as e:
                print("Error in task execution: ", e)
            except Exception as e:
                print(f"Other Error: {e}")
                
        
        print("Performance matrix shape: ", len(performance_matrix), len(performance_matrix[0]))


        # configuration_cell:ConfigurationCell = PerformanceMatrix.process_task(tasks[34],configspace,pipeline_registry,budget = None, seed = 42)
        # performance_matrix[configuration_cell.row][configuration_cell.col] = configuration_cell
        # self.write_performace_matrix(performance_matrix)


def main():
    config = Config()
    performance_matrix = PerformanceMatrix(f"{config.get_results_path()}/best_candidate_runs.json", config.get_train_path(),
                                          f"{config.get_results_path()}/performance_matrix_runs.json")
    performance_matrix.create_performance_matrix()

if __name__ == "__main__":
    main()

        



       

    