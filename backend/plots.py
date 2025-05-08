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

    def compare_results(self, framework_test_accuracies, framework_meta_accuracies):
        count = 0
        count2 = 0
        for i in range(len(framework_test_accuracies)):
            if framework_test_accuracies[i] < framework_meta_accuracies[i]:
                count += 1
            elif (framework_test_accuracies[i] - framework_meta_accuracies[i]) <= 0.01:
                count2 += 1
        print("Fraction of datasets where meta-learning improved accuracy: ", count/len(framework_test_accuracies))
        print("Fraction of datasets where meta-learning is less than 1\% difference in accuracy : ", count2/len(framework_test_accuracies))

    
    def set_fonts(self):
        plt.rcParams['font.family'] = 'DejaVu Sans'
        # plt.rcParams['font.size'] = 24  # Default font size for everything
        plt.rcParams['axes.titlesize'] = 24  # Title size
        plt.rcParams['axes.labelsize'] = 16  # Axis label size
        plt.rcParams['xtick.labelsize'] = 16  # x-axis tick size
        plt.rcParams['ytick.labelsize'] = 16  # y-axis tick size
        plt.rcParams['legend.fontsize'] = 16 # Legend font size     
        
    
    def plot_autosklearn(self):
        framework_dataset_ids, framework_test_accuracies = self.extract_results(self.framework_results)
        autosklearn_dataset_ids, autosklearn_test_accuracies = self.extract_results(self.autosklearn_results)
        self.compare_results(framework_test_accuracies, autosklearn_test_accuracies)
        X_values = [i for i in range(1,len(framework_dataset_ids)+1)]
       
        plt.figure(figsize=(12, 6))
        self.set_fonts()
        plt.plot(X_values, framework_test_accuracies,label = "Proposed Framework",linestyle='-',marker='o', color = '#f95d6a')
        plt.plot(X_values, autosklearn_test_accuracies,label = "Auto-sklearn",linestyle='-',marker='o',color = "#3f3cbb")
        plt.xticks(X_values)
        plt.xlabel("Dataset ID")
        plt.ylabel("Accuracy")
        plt.title("Test Accuracy: Proposed Framework vs Auto-sklearn")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.filename)
    
    def plot(self):
        framework_dataset_ids, framework_test_accuracies = self.extract_results(self.framework_results)
        autosklearn_dataset_ids, autosklearn_test_accuracies = self.extract_results(self.autosklearn_results)
        self.compare_results(framework_test_accuracies, autosklearn_test_accuracies)
        X_values = [i for i in range(1,len(framework_dataset_ids)+1)]
        plt.figure(figsize=(12, 6))
        self.set_fonts()
        plt.plot(X_values, framework_test_accuracies,label = "Without Meta-Learning",linestyle='-',marker='o', color = '#d7263d')
        plt.plot(X_values, autosklearn_test_accuracies,label = "With Meta-Learning",linestyle='-',marker='o',color = "#003f5c")
        plt.xticks(X_values)
        plt.xlabel("Dataset ID")
        plt.ylabel("Accuracy")
        plt.title("Test Accuracy: Without vs With Meta-Learning")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.filename)

def main():
    config = Config()
    # plots = Plots(config.get_results_path() + "/framework_benchmark_runs.json",
    #               config.get_results_path() + "/framework_benchmark_runs_with_meta_learning.json",  
    #               config.get_results_path() + "/benchmark_runs.png")
    # plots.plot()
    plots = Plots(config.get_results_path() + "/framework_benchmark_runs_with_meta_learning.json",
                  config.get_results_path() + "/autosklearn_benchmark_runs.json",  
                  config.get_results_path() + "/autosklearn_benchmark_runs.png")
    plots.plot_autosklearn()
    

if __name__ == "__main__":
    main()



