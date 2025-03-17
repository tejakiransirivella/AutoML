import openml
from sklearn.impute import SimpleImputer
from backend.config import Config
import pandas as pd

config = Config()

def impute_missing_values(dataset:openml.OpenMLDataset):
    X,_,_,_ = dataset.get_data(dataset_format="dataframe")
    print("Before imputation:")
    print(X)
    X_numeric = X.select_dtypes(include=["number"])
    X_categorical = X.select_dtypes(include=["category"])

    if not X_numeric.empty:
       numeric_imputer = SimpleImputer(strategy="mean")
       X_numeric = pd.DataFrame(numeric_imputer.fit_transform(X_numeric), columns=X_numeric.columns)

    if not X_categorical.empty:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        X_categorical = pd.DataFrame(categorical_imputer.fit_transform(X_categorical), columns=X_categorical.columns)
    
    X = pd.concat([X_numeric, X_categorical], axis=1)
    
def check_missing_values(dataset:openml.OpenMLDataset):
    if dataset.qualities["PercentageOfMissingValues"] > 0:
         impute_missing_values(dataset)
  

def is_string_or_time(dataset:openml.OpenMLDataset):
    for feature in dataset.features.values():
        if feature.data_type in ["string", "time", "date"]:
            return True
    return False

def is_sparse(dataset:openml.OpenMLDataset):
    X,_,_,_ = dataset.get_data(dataset_format="dataframe")
    X_numeric = X.select_dtypes(exclude=["category", "object"])
    zeroes = (X_numeric == 0).sum().sum()
    return zeroes > 0.5 * X_numeric.size

    
def is_valid_features(dataset: openml.OpenMLDataset):
    num_features = dataset.qualities["NumberOfFeatures"]
    return num_features >= config.get_min_features()


def is_valid_rows(dataset:openml.OpenMLDataset):
    num_rows = dataset.qualities["NumberOfInstances"]
    return num_rows >= config.get_min_rows() and num_rows <= config.get_max_rows()



def is_valid_dataset(dataset:openml.OpenMLDataset):
   
    return  is_valid_rows(dataset) and \
            is_valid_features(dataset) and \
            not is_string_or_time(dataset) and \
            not is_sparse(dataset)


def collect_datasets():
    suite = openml.study.get_suite("OpenML-CC18")
    dataset_ids = suite.data
    filtered_datasets = []
    for dataset_id in dataset_ids:
        dataset = openml.datasets.get_dataset(dataset_id)
        if is_valid_dataset(dataset):
            filtered_datasets.append(dataset)
    print(len(filtered_datasets))


def test():
    suite = openml.study.get_suite("OpenML-CC18")
    # features_types = set()
    dataset_ids = suite.data
    # for dataset_id in dataset_ids:
    #     dataset = openml.datasets.get_dataset(dataset_id)
    #     for feature in dataset.features.values():
    #         features_types.add(feature.data_type)
    # print(features_types)
    # dataset = openml.datasets.get_dataset(1468)
    # X,y,_,_ = dataset.get_data(dataset_format="dataframe")
    # X_numeric = X.select_dtypes(exclude=["category", "object"])
    # zeroes = (X_numeric == 0).sum().sum()
    # print(zeroes)
    # print(X_numeric.size)
    for dataset_id in dataset_ids:
        dataset = openml.datasets.get_dataset(dataset_id)
        check_missing_values(dataset)


if __name__ == "__main__":
    # collect_datasets()
    test()


