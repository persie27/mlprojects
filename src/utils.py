import sys
import os

from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import dill

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report ={}

        for i in range(0, len(models)):
            model = list(models.values())[i]
            # logging.info(f"Model name: {model}")
            # logging.info(f"Utils  **** X_train {X_train[:1,:]}")
            model.fit(X_train, y_train)

            y_hat_train = model.predict(X_train)
            y_hat_test = model.predict(X_test)

            train_model_score = r2_score(y_hat_train, y_train)
            test_model_score = r2_score(y_hat_test, y_test)

            report[list(models.keys())[i]] = test_model_score
        
        return report
    except Exception as e:
        raise CustomException(e,sys)