class BestCandidateRun:

    def __init__(self,dataset_id:int, best_config:dict, val_score:float,test_score:float):
        self.dataset_id = dataset_id
        self.best_config = best_config
        self.val_score = val_score
        self.test_score = test_score
       