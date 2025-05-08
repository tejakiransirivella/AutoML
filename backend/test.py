import openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from backend.Optimizer import Optimizer
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
import backend.meta_learning.preprocess as preprocess
from config import Config
import numpy as np

def test(max_iter:int=100):
    print(max_iter)

def main(budget:int=None):
    test(**({"max_iter": budget} if budget is not None else {}))

main(1000)