import openml
from backend.config import Config
import sklearn.ensemble as se


se.ExtraTreesClassifier()
config = Config()

def main():
    print(openml.datasets.get_dataset(3).name)
    tasks = openml.tasks.list_tasks(output_format="dataframe",data_id=1486,
                                    task_type=openml.tasks.TaskType.SUPERVISED_CLASSIFICATION)
    print(tasks.index.to_list())

    supported_models = ['RandomForest', 
                    'DecisionTree',
                    'LogisticRegression'] 
    
    flows = openml.flows.list_flows(output_format="dataframe")
    i = 0
    for name in set(flows["name"]):
        if name.startswith("sklearn"):
            print(name)
            i +=1
        if i == 20:
            break


   
   

if __name__ == "__main__":
    main()