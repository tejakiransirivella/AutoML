import json

class Config:
    def __init__(self):
       self.config =  json.load(open("backend/config.json"))

    def get_openml_api_key(self):
        return self.config["openml_api_key"]
    
    def get_min_rows(self):
        return self.config["min_rows"]
    
    def get_max_rows(self):
        return self.config["max_rows"]
    
    def get_min_features(self):
        return self.config["min_features"]