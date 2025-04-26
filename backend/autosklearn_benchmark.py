import json
import pathlib
import openml
import pandas as pd
import ray
import ray.exceptions
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from backend.config import Config
from backend.meta_learning.preprocess import Preprocess
from backend.meta_learning.best_candidate_run import BestCandidateRun

from autosklearn.classification import AutoSklearnClassifier

config = Config()

class AutosklearnBenchmark(Preprocess):
    
    def __init__(self,filename,data_path):
        Preprocess.__init__(self,data_path)
        ray.init(address="auto") 
        self.dataset_ids = []
        self.nodes = len(ray.nodes())
        self.filename = filename
        self.data_path = data_path

    def write_to_file(self, best_candidate_runs, filename):
        with open(filename, "w") as f:
            json.dump({"best_candidates":[run.__dict__ for run in best_candidate_runs]},f,indent=4, default=str)
            
    @staticmethod
    @ray.remote
    def process_task(task):
        best_candidate_run = None
        autoclassifier = AutoSklearnClassifier(
                        time_left_for_this_task=3600,
                        seed=42,
                        memory_limit=None
                    )

        try:
            dataset_id,X,y = task
            print("dataset id : ", dataset_id)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            autoclassifier.fit(X_train,y_train)
            print("Starting prediction")
            y_pred = autoclassifier.predict(X_test)
            test_accuracy = accuracy_score(y_test ,y_pred)
            print("Test accuracy: ", test_accuracy)
            results = pd.DataFrame(autoclassifier.cv_results_)
            best_index = results['rank_test_scores'].argmin()
            val_score = results.loc[best_index, 'mean_test_score']
            best_candidate_run = BestCandidateRun(dataset_id=dataset_id, best_config={}, 
                                                val_score=val_score, test_score=test_accuracy)
        except Exception as e:
            print(f"Error processing dataset {task[0]}: {e}")
            
        return best_candidate_run
    
    def find_best_candidate(self):
        tasks = []
        # self.dataset_ids = self.dataset_ids[0:1]
        for dataset_id in self.dataset_ids:
            dataset = self.load_dataset(dataset_id)
            target = openml.datasets.get_dataset(dataset_id).default_target_attribute
            y = dataset[target]
            X = dataset.drop(columns=[target])

            tasks.append((dataset_id,X, y))
        
        print("no of tasks: ", len(tasks))
        best_candidate_runs = []
        futures = []
        for task in tasks:
            future =  AutosklearnBenchmark.process_task.remote(task)
            futures.append(future)
        
        try:
            best_candidate_runs = ray.get(futures)
        except ray.exceptions.RayTaskError as e:
            print("Error in task execution: ", e)
        self.write_to_file(best_candidate_runs, self.filename)
    
    def collect_dataset_ids(self):
        data = pathlib.Path(self.data_path)
        for f in data.iterdir():
            if f.is_file() and f.suffix == ".csv":
                dataset_id = int(f.stem)
                self.dataset_ids.append(dataset_id)

    
def main():
    config = Config()
    benchmark = AutosklearnBenchmark("autosklearn_benchmark_runs.json", config.get_test_path())
    # benchmark.collect_datasets()
    benchmark.collect_dataset_ids()
    benchmark.find_best_candidate()
    # benchmark.test()

main()

        