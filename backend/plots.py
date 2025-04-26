import json
import matplotlib.pyplot as plt
from backend.config import Config


class Plots:
    def __init__(self,framework_results, autosklearn_results, filename):
        self.framework_results = framework_results
        self.autosklearn_results = autosklearn_results
        self.filename = filename
        
    def extract_results(self, results_path):
        dataset_ids = []
        test_accuracies = []
        
        with open(results_path, "r") as f:
            data = json.load(f)
            for dataset in data["best_candidates"]:
                dataset_ids.append(dataset["dataset_id"])
                test_accuracies.append(dataset["test_score"])
            dataset_ids, test_accuracies = zip(*sorted(zip(dataset_ids,  test_accuracies)))
            return dataset_ids, test_accuracies

    def plot(self):
        framework_dataset_ids, framework_test_accuracies = self.extract_results(self.framework_results)
        autosklearn_dataset_ids, autosklearn_test_accuracies = self.extract_results(self.autosklearn_results)
        X_values = [i for i in range(1,len(framework_dataset_ids)+1)]
        plt.figure(figsize=(12, 6))
        plt.plot(X_values, framework_test_accuracies,label = "My Framework",linestyle='-',marker='o', color = 'green')
        plt.plot(X_values, autosklearn_test_accuracies,label = "AutoSklearn",linestyle='-',marker='o')
        plt.xticks(X_values)
        plt.xlabel("Dataset ID")
        plt.ylabel("Accuracy")
        plt.title("Accuracy against test datasets for My Framework vs AutoSklearn")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.filename)

def main():
    config = Config()
    plots = Plots(config.get_results_path() + "/myframework_benchmark_runs.json",
                  config.get_results_path() + "/autosklearn_benchmark_runs.json",  
                  config.get_results_path() + "/benchmark_runs.png")
    plots.plot()

if __name__ == "__main__":
    main()



