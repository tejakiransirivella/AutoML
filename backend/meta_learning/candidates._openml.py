import openml
from backend.config import Config
import preprocess
import openml.tasks.task as task1
import ray


ray.init(address="auto")


class CandidatePipeline:


    def __init__(self,dataset_id:int):  
        self.dataset_id = dataset_id
        self.best_candidate = None
        self.models = ["sklearn.ensemble.RandomForestClassifier",
                       "sklearn.ensemble.ExtraTreesClassifier",
                       "sklearn.ensemble.HistGradientBoostingClassifier",
                       "sklearn.neural_network.MLPClassifier",
                       "sklearn.linear_model.PassiveAggressiveClassifier",
                       "sklearn.linear_model.SGDClassifier"]
        self.nodes = len(ray.nodes())

    def write_to_file(self, unique_models, filename):
        with open(filename, "w") as f:
            for model in unique_models:
                f.write(model + "\n")
    
    @staticmethod
    @ray.remote
    def process_runs(runs, models):
        counter = 0
        unique_models = set()
        for _, run in runs.iterrows():
            flow = openml.flows.get_flow(run["flow_id"])
            if flow.name.startswith("sklearn"):
                unique_models.add(flow.name)
            # if flow.name in models and \
            #     run["task_evaluation_measure"] == "predictive_accuracy":
            counter += 1
        return unique_models
                
    def find_best_candidate(self):
        tasks = openml.tasks.list_tasks(data_id=self.dataset_id,task_type=task1.TaskType.SUPERVISED_CLASSIFICATION,
                          output_format="dataframe")
        print("no of tasks: ", len(tasks))
        unique_models = set()
        chunks = []
        for _, task in tasks.iterrows():
            print("task id: ", task["tid"])
            runs = openml.runs.list_runs(task=[task["tid"]], output_format="dataframe")
            print("no of runs: ", len(runs))
            i = 0
            if len(runs) >= self.nodes:
                chunk_size = len(runs) // self.nodes
                for i in range(0,len(runs),chunk_size):
                    chunk = runs[i:i+chunk_size]
                    chunks.append(chunk)
            if len(runs[i:]) > 0:
                chunks[-1] = runs[i:]
        print("no of chunks: ", len(chunks))
        futures = []
        for i in range(self.nodes):
            if i < len(chunks):
                future = CandidatePipeline.process_runs.remote(chunks[i], self.models)
                futures.append(future)
        results = ray.get(futures)
        for result in results:
            unique_models.update(result)
        print("no of unique models: ", len(unique_models))
        self.write_to_file(unique_models, "unique_models.txt")
        # print("results: ", results)
           
        # self.write_to_file(unique_models, "unique_models.txt")




def main():
    candidate = CandidatePipeline(3)
    candidate.find_best_candidate()
 
      


 
   

if __name__ == "__main__":
    main()