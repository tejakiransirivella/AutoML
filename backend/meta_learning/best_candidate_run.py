class BestCandidateRun:

    def __init__(self,dataset_id:int=None, best_config:dict=None, val_score:float=None,test_score:float=None):
        self.dataset_id = dataset_id
        self.best_config = best_config
        self.val_score = val_score
        self.test_score = test_score
       