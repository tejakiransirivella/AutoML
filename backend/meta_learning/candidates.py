import ray
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from backend.autoclassifier import AutoClassifier
import backend.meta_learning.preprocess as preprocess
from backend.meta_learning.best_candidate_run import BestCandidateRun
import json
import pathlib


ray.init(address="auto")

class CandidatePipeline:

    def __init__(self):  
        self.dataset_ids = []
        self.best_candidate = None
        self.validation_score = None
        self.nodes = len(ray.nodes())

    def write_to_file(self, best_candidate_runs, filename):
        with open(filename, "w") as f:
            for run in best_candidate_runs:
                json.dump(run.__dict__,f,indent=4)
            

    @staticmethod
    @ray.remote
    def process_task(task,autoclassifier: AutoClassifier):
        dataset_id,X,y = task
        X,y = autoclassifier.one_hot_encoding(X,y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        autoclassifier.fit(X_train,y_train)
        y_pred = autoclassifier.predict(X_test)
        test_accuracy = accuracy_score(y_test ,y_pred)
        best_candidate_run = BestCandidateRun(dataset_id=dataset_id, best_config=autoclassifier.best_config, 
                                              val_score=autoclassifier.val_score, test_score=test_accuracy)
        return best_candidate_run
    
    def find_best_candidate(self):
        tasks = []
        autoclassifier = AutoClassifier(seed=42, walltime_limit=60, min_budget=10, max_budget=1000)
        for dataset_id in self.dataset_ids:
            dataset = preprocess.load_dataset(dataset_id)
            print("dataset id : ", dataset_id)
            y = dataset.iloc[:,-1]
            X = dataset.iloc[:,:-1]
            tasks.append((dataset_id,X, y))
        
        print("no of tasks: ", len(tasks))
        best_candidate_runs = []
        futures = []
        for task in tasks:
            future =  CandidatePipeline.process_task.remote(task, autoclassifier)
            futures.append(future)
        
        try:
            best_candidate_runs = ray.get(futures)
            pass
        except ray.exceptions.RayTaskError as e:
            print("Error in task execution: ", e)
        self.write_to_file(best_candidate_runs, "best_candidate_runs.json")

    def collect_dataset_ids(self):
        current_file = pathlib.Path(__file__).resolve(strict=True)
        project_root = current_file.parents[2]
        train_data = project_root / "data" / "train"
        for f in train_data.iterdir():
            if f.is_file() and f.suffix == ".csv":
                dataset_id = int(f.stem)
                self.dataset_ids.append(dataset_id)

def main():
    candidate = CandidatePipeline()
    candidate.collect_dataset_ids()
    candidate.find_best_candidate()
    # print("Dataset ids: ", candidate.dataset_ids)

if __name__ == "__main__":
    main()
    

        
       
        
        

            


