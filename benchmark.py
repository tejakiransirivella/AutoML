import json
import pathlib
import openml
import pandas as pd
import ray
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from backend.config import Config
from backend.meta_learning.preprocess import Preprocess
from backend.autoclassifier import AutoClassifier
from backend.meta_learning.best_candidate_run import BestCandidateRun
from backend.meta_learning.candidates import CandidatePipeline

config = Config()

ray.init(address="auto")

class Benchmark(Preprocess,CandidatePipeline):

    def __init__(self,filename,data_path):
        self.dataset_ids = []
        self.nodes = len(ray.nodes())
        self.filename = filename
        self.data_path = data_path
        super().__init__(data_path)

    def filter_datasets(self,dataset_ids):
        print("Filtering datasets...")
        filtered_ids = []
        for dataset_id in dataset_ids:
            dataset = openml.datasets.get_dataset(dataset_id)
            X,_,_,_ = dataset.get_data(dataset_format="dataframe")
            if not any(X.dtypes == "object"):
                filtered_ids.append(dataset_id)
        self.dataset_ids = filtered_ids
        return filtered_ids

    def collect_datasets(self):
        suite = openml.study.get_suite(218)
        for task_id in suite.tasks:
            task = openml.tasks.get_task(task_id)
            self.dataset_ids.append(task.dataset_id)
        print("Total datasets:", len(self.dataset_ids))
        filtered_dataset_ids = self.filter_datasets(self.dataset_ids)
        print("Filtered datasets:", len(filtered_dataset_ids))
        self.preprocess_datasets(filtered_dataset_ids)


def main():
    config = Config()
    benchmark = Benchmark("test_datasets_runs.json", config.get_test_path())
    benchmark.collect_datasets()
    benchmark.collect_dataset_ids()
    benchmark.find_best_candidate()


if __name__ == "__main__":
    main()

