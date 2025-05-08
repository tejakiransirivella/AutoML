import numpy as np
import openml
from sklearn.impute import SimpleImputer
from backend.config import Config
import pandas as pd
import os

config = Config()

class Preprocess:

    def __init__(self,data_path):
        self.data_path = data_path

    def load_dataset(self,dataset_id):
        if os.path.exists(f"{self.data_path}/{dataset_id}.csv"):
            return pd.read_csv(f"{self.data_path}/{dataset_id}.csv")
        openml_dataset = openml.datasets.get_dataset(dataset_id)
        X, _, _, _ = openml_dataset.get_data(dataset_format="dataframe")
        self.save_dataset(X, dataset_id)
        return X

    def save_dataset(self,data:pd.DataFrame,dataset_id):
        data.to_csv(f"{self.data_path}/{dataset_id}.csv", index=False)

    def impute_missing_values(self,dataset:openml.OpenMLDataset):
        X,_,_,_ = dataset.get_data(dataset_format="dataframe")
        X.replace(['NaN', 'nan', 'null', 'None', 'missing', ''], np.nan, inplace=True)
        print("Before imputation:" , dataset.dataset_id)

        # handle boolean columns
        columns = X.select_dtypes(include=["bool"]).columns
        for column in columns:
            X[column] = X[column].astype("category")
        
        columns = X.columns.to_list()
        X_numeric = X.select_dtypes(include=["number"])
        X_categorical = X.select_dtypes(include=["category","bool"])


        if not X_numeric.empty:
            numeric_imputer = SimpleImputer(strategy="mean")
            X_numeric = pd.DataFrame(numeric_imputer.fit_transform(X_numeric), columns=X_numeric.columns)

        if not X_categorical.empty:
            categorical_imputer = SimpleImputer(strategy="most_frequent")
            X_categorical = pd.DataFrame(categorical_imputer.fit_transform(X_categorical), columns=X_categorical.columns)
        
        X = pd.concat([X_numeric, X_categorical], axis=1)

        X = X[columns]

        return X
        
    def check_missing_values(self,dataset:openml.OpenMLDataset):
        return True
    
    def is_string_time_object(self,dataset:openml.OpenMLDataset):
        for feature in dataset.features.values():
            if feature.data_type in ["string", "time", "date"]:
                return True
            
        X,_,_,_ = dataset.get_data(dataset_format="dataframe")
        if any(X.dtypes == "object"):
            return True
        
        return False

    def is_sparse(self,dataset:openml.OpenMLDataset):
        X,_,_,_ = dataset.get_data(dataset_format="dataframe")
        X_numeric = X.select_dtypes(exclude=["category", "object"])
        zeroes = (X_numeric == 0).sum().sum()
        return zeroes > 0.5 * X_numeric.size

        
    def is_valid_features(self,dataset: openml.OpenMLDataset):
        num_features = dataset.qualities["NumberOfFeatures"]
        return num_features >= config.get_min_features()


    def is_valid_rows(self,dataset:openml.OpenMLDataset):
        num_rows = dataset.qualities["NumberOfInstances"]
        return num_rows >= config.get_min_rows() and num_rows <= config.get_max_rows()

    def is_valid_dataset(self,dataset:openml.OpenMLDataset):
    
        return  self.is_valid_rows(dataset) and \
                self.is_valid_features(dataset) and \
                not self.is_string_time_object(dataset) and \
                not self.is_sparse(dataset)


    def preprocess_dataset(self,dataset:openml.OpenMLDataset):
        if self.check_missing_values(dataset):
            return self.impute_missing_values(dataset)
        X,_,_,_ = dataset.get_data(dataset_format="dataframe")
        return X

    def preprocess_datasets(self,dataset_ids):
        print("Preprocessing datasets...")
        for dataset_id in dataset_ids:
            print("Preprocessing dataset id : ", dataset_id)
            dataset = openml.datasets.get_dataset(dataset_id)
            data = self.preprocess_dataset(dataset)
            self.save_dataset(data, dataset_id)

    def filter_datasets(self,dataset_ids):
        print("Filtering datasets...")
        filtered_dataset_ids = []
        for dataset_id in dataset_ids:
            dataset = openml.datasets.get_dataset(dataset_id)
            if self.is_valid_dataset(dataset):
                filtered_dataset_ids.append(dataset_id)
        return filtered_dataset_ids
                
    def collect_datasets(self):
        suite = openml.study.get_suite("OpenML-CC18")
        dataset_ids = suite.data
        print("Total datasets:", len(dataset_ids))
        filtered_dataset_ids = self.filter_datasets(dataset_ids)
        print("Filtered datasets:", len(filtered_dataset_ids))
        self.preprocess_datasets(filtered_dataset_ids)


def test():
    preprocess = Preprocess("/home/stu11/s15/ts7244/capstone/AutoML/data/test")
    # preprocess.collect_datasets()
   
    # suite = openml.study.get_suite("OpenML-CC18")
    # features_types = set()
    # dataset_ids = suite.data
    # for dataset_id in dataset_ids:
    #     dataset = openml.datasets.get_dataset(dataset_id)
    #     for feature in dataset.features.values():
    #         features_types.add(feature.data_type)
    # print(features_types)
    dataset = openml.datasets.get_dataset(1067)
    preprocess  = Preprocess(config.get_test_path())
    preprocess.preprocess_dataset(dataset)

    # print( dataset.qualities["PercentageOfMissingValues"] > 0)
    # X = preprocess.load_dataset(41147)
    # X.replace(['NaN', 'nan', 'null', 'None', 'missing', ''], np.nan, inplace=True)
    # print(X.head())
    # print(X.isnull().values.any())
    X = preprocess.impute_missing_values(dataset)
   
    preprocess.save_dataset(X, 41147)
    # print(dataset.name)
    # X = preprocess.load_dataset(40927)
    
    # # X,_,_,_ = dataset.get_data(dataset_format="dataframe")
    # print(len(X))
    # print(len(X.columns))
    # X_numeric = X.select_dtypes(exclude=["category", "object"])
    # zeroes = (X_numeric == 0).sum().sum()
    # print(zeroes)
    # print(X_numeric.size)
    # for dataset_id in dataset_ids:
    #     dataset = openml.datasets.get_dataset(dataset_id)
    #     check_missing_values(dataset)

    # dataset = openml.datasets.get_dataset(1053)
    # impute_missing_values(dataset)


if __name__ == "__main__":
    # collect_datasets()
    test()


