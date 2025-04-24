import openml

from backend.config import Config
from backend.meta_learning.preprocess import Preprocess
from backend.meta_learning.candidates import CandidatePipeline

config = Config()

class Benchmark(Preprocess,CandidatePipeline):

    def __init__(self,filename,data_path):
        Preprocess.__init__(self,data_path)
        CandidatePipeline.__init__(self, filename,data_path)

    def collect_datasets(self):
        suite = openml.study.get_suite(218)
        for task_id in suite.tasks:
            task = openml.tasks.get_task(task_id)
            self.dataset_ids.append(task.dataset_id)
        print("Total datasets:", len(self.dataset_ids))
        filtered_dataset_ids = self.filter_datasets(self.dataset_ids)
        self.dataset_ids = filtered_dataset_ids
        print("Filtered datasets:", len(filtered_dataset_ids))
        self.preprocess_datasets(filtered_dataset_ids)

    def test(self):
        self.collect_dataset_ids()
        for dataset_id in self.dataset_ids:
            dataset = self.preprocess.load_dataset(dataset_id)
             


def main():
    config = Config()
    benchmark = Benchmark("test_datasets_runs.json", config.get_test_path())
    # benchmark.collect_datasets()
    benchmark.collect_dataset_ids()
    benchmark.find_best_candidate()
    # benchmark.test()


if __name__ == "__main__":
    main()

