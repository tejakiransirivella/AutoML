import json
from backend.meta_learning.performance_matrix import ConfigurationCell
from backend.config import Config
from performance_matrix import PerformanceMatrix


class Portfolio:
    def __init__(self,performance_matrix_path:str=None, best_candidates_path:str=None, data_path:str=None,portfolio_path:str=None):
        self.performance_matrix_path = performance_matrix_path
        self.performance_matrix = PerformanceMatrix(best_candidates_path=best_candidates_path,
        data_path=data_path,performance_matrix_path=performance_matrix_path)
        self.portfolio_path = portfolio_path
        self.portfolio = []
        self.portfolio_size = 25

    def load_performance_matrix(self):
        performance_matrix = {}
        with open(self.performance_matrix_path, "r") as f:
            data = json.load(f)
            cells = data["performace_matrix"]
            print(len(cells))
            for cell in cells:
                config = cell["config"]
                config_key = tuple(config.items())
                dataset_id = cell["dataset_id"]
                configuration_cell: ConfigurationCell = ConfigurationCell(
                                                    config=config,
                                                    dataset_id=dataset_id,
                                                    row=cell["row"],
                                                    col=cell["col"],
                                                    val_score=cell["val_score"])
                
                if config_key not in performance_matrix:
                    performance_matrix[config_key] = {}
                performance_matrix[config_key][dataset_id] = configuration_cell

        return performance_matrix
    
    def get_estimated_acc_dataset(self,configuration, dataset_id, performance_matrix):
        best_acc = 0
        for config in self.portfolio:
            best_acc = max(best_acc,performance_matrix[configuration][dataset_id].val_score)
        
        best_acc = max(best_acc,performance_matrix[configuration][dataset_id].val_score)
        return best_acc
    
    def get_estimated_acc_datasets(self, configuration, dataset_ids, performance_matrix):
        est_acc = 0
        for dataset_id in dataset_ids:
            est_acc += self.get_estimated_acc_dataset(configuration,dataset_id,performance_matrix)
        return est_acc

    def build_portfolio(self):
        dataset_ids,configurations,_ = self.performance_matrix.load_configurations()
        performance_matrix = self.load_performance_matrix()

        unselected_configurations = set()
        for configuration in configurations:
            config_key = tuple(configuration.items())
            unselected_configurations.add(config_key)

        # greedy algorithm to build portfolio
        for _ in range(self.portfolio_size):
            best_configuration = None
            best_score_seen = -float("inf")
            for configuration in unselected_configurations:
                est_acc = self.get_estimated_acc_datasets(configuration,dataset_ids,performance_matrix)
                if est_acc > best_score_seen:
                    best_configuration = configuration
                    best_score_seen = est_acc
            if best_configuration is None:
                    break
            self.portfolio.append(best_configuration)
            unselected_configurations.remove(best_configuration)

        portfolio = []
        for config_key in self.portfolio:
            configuration = dict(config_key)
            portfolio.append(configuration)
        self.portfolio = portfolio
        print("Portfolio: ", len(portfolio))
   

    def write_to_file(self):
        with open(self.portfolio_path, "w") as f:
            json.dump(
                {"portfolio":[
                    configuration
                    for configuration in self.portfolio
                    ]
                },
                f,indent=4, default=str) 
def main():
     config = Config()
     portfolio = Portfolio(
        performance_matrix_path=f"{config.get_results_path()}/performance_matrix_runs.json",
        best_candidates_path=f"{config.get_results_path()}/best_candidate_runs.json",
        data_path=config.get_train_path(),
        portfolio_path=f"{config.get_results_path()}/portfolio.json")
     
     portfolio.build_portfolio()
     portfolio.write_to_file()

if __name__ == "__main__":
    main()