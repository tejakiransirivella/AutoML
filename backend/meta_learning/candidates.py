import ray
import ray.exceptions
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from backend.autoclassifier import AutoClassifier
from backend.meta_learning.preprocess import Preprocess
from backend.meta_learning.best_candidate_run import BestCandidateRun
from backend.config import Config

import json
import pathlib
import openml

class CandidatePipeline:

    def __init__(self, filename,data_path): 
        ray.init(address="auto") 
        self.dataset_ids = []
        self.nodes = len(ray.nodes())
        self.filename = filename
        self.data_path = data_path
        self.preprocess = Preprocess(data_path)

    def write_to_file(self, best_candidate_runs, filename):
        with open(filename, "w") as f:
            json.dump({"best_candidates":[run.__dict__ for run in best_candidate_runs]},f,indent=4, default=str)
            

    @staticmethod
    @ray.remote
    def process_task(task,autoclassifier: AutoClassifier):
        best_candidate_run = None
        try:
            dataset_id,X,y = task
            print("dataset id : ", dataset_id)
            X,y = autoclassifier.one_hot_encoding(X,y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            autoclassifier.fit(X_train,y_train)
            print("Starting prediction")
            y_pred = autoclassifier.predict(X_test)
            test_accuracy = accuracy_score(y_test ,y_pred)
            print("Test accuracy: ", test_accuracy)
            best_candidate_run = BestCandidateRun(dataset_id=dataset_id, best_config=dict(autoclassifier.best_config), 
                                                val_score=autoclassifier.val_score, test_score=test_accuracy)
        except Exception as e:
            print(f"Error processing dataset {task[0]}: {e}")
            
        return best_candidate_run
    
    def find_best_candidate(self):
        tasks = []
        # self.dataset_ids = self.dataset_ids[0:1]
        for dataset_id in self.dataset_ids:
            dataset = self.preprocess.load_dataset(dataset_id)
            target = openml.datasets.get_dataset(dataset_id).default_target_attribute
            print("dataset id : ", dataset_id, "target: ", target)
            y = dataset[target]
            X = dataset.drop(columns=[target])

            tasks.append((dataset_id,X, y))
        
        print("no of tasks: ", len(tasks))
        best_candidate_runs = []
        futures = []
        for task in tasks:
            autoclassifier = AutoClassifier(seed=42, walltime_limit=600, min_budget=10, max_budget=500)
            future =  CandidatePipeline.process_task.remote(task, autoclassifier)
            futures.append(future)
        
        for future in futures:
            try:
                best_candidate_run = ray.get(future)
                if best_candidate_run is not None:
                    best_candidate_runs.append(best_candidate_run)
            except ray.exceptions.RayTaskError as e:
                print("Error in task execution: ", e)
            except Exception as e:
                print(f"Other Error: {e}")
                
        print(f"Successfully collected {len(best_candidate_runs)} runs out of {len(futures)} datasets.")
        self.write_to_file(best_candidate_runs, self.filename)

    def collect_dataset_ids(self):
        train_data = pathlib.Path(self.data_path)
        for f in train_data.iterdir():
            if f.is_file() and f.suffix == ".csv":
                dataset_id = int(f.stem)
                self.dataset_ids.append(dataset_id)

def main():
    config = Config()
    candidate = CandidatePipeline("best_candidate_runs.json", config.get_train_path())
    candidate.collect_dataset_ids()
    candidate.find_best_candidate()
    # print("Dataset ids: ", candidate.dataset_ids)

if __name__ == "__main__":
    main()
    

        
       
        
        

            


