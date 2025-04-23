import json
import pathlib

class Config:
    def __init__(self):
       self.config =  json.load(open("backend/config.json"))
       self.path = pathlib.Path(__file__).resolve().parents[1]
       self.config["train_path"] = str(self.path) + "/" + "data" + "/" + "train"
       self.config["test_path"] = str(self.path) + "/" + "data" + "/" + "test"

    def get_openml_api_key(self):
        return self.config["openml_api_key"]
    
    def get_min_rows(self):
        return self.config["min_rows"]
    
    def get_max_rows(self):
        return self.config["max_rows"]
    
    def get_min_features(self):
        return self.config["min_features"]
    
    def get_train_path(self):
        return self.config["train_path"]
    
    def get_test_path(self):
        return self.config["test_path"]