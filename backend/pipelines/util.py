from ConfigSpace import Configuration
import numpy as np

def get_config_for_model(model_name:str,config: Configuration):
    config_dict = dict(config)
    res = {}
    for key in config_dict.keys():
        if key.startswith(model_name):
            new_key = key.replace(f"{model_name}.", "")
            res[new_key] = config_dict[key]
            if res[new_key] == "True":
                res[new_key] = True
            elif res[new_key] == "False":
                res[new_key] = False
    
    for key,value in res.items():
        if isinstance(value,(np.generic,np.bool)):
            res[key] = value.item()
            
    # print(res)
    return res