import openml
from backend.config import Config
import preprocess
import openml.tasks.task as task1


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

    def write_to_file(self, unique_models, filename):
        with open(filename, "w") as f:
            for model in unique_models:
                f.write(model + "\n")
    
    def find_best_candidate(self):
        tasks = openml.tasks.list_tasks(data_id=self.dataset_id,task_type=task1.TaskType.SUPERVISED_CLASSIFICATION,
                          output_format="dataframe")
        print("no of tasks: ", len(tasks))
        unique_models = set()
        for _, task in tasks.iterrows():
            print("task id: ", task["tid"])
            runs = openml.runs.list_runs(task=[task["tid"]], output_format="dataframe")
            print("no of runs: ", len(runs))
            for _, run in runs.iterrows():
                # print("       run id: ", run["run_id"])
                flow = openml.flows.get_flow(run["flow_id"])
                unique_models.add(flow.name)
                # print(flow.name)
                if flow.name in self.models and \
                    run["task_evaluation_measure"] == "predictive_accuracy":
                    print(run.evaluations)
                    break
            break
        self.write_to_file(unique_models, "unique_models.txt")




def main():
    candidate = CandidatePipeline(3)
    candidate.find_best_candidate()
 
      


 
   

if __name__ == "__main__":
    main()