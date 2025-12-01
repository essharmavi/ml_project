import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(error_message=e, error_detail=sys)


def evaluate_model(X_train, Y_train, X_test, Y_test, models):
    try:
        models_report = {}
        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            model.fit(X_train, Y_train)

            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)

            r2_score_train = r2_score(Y_train, Y_train_pred)
            r2_score_test = r2_score(Y_test, Y_test_pred)

            models_report[model_name] = r2_score_test
        
        return models_report

    except Exception as e:
        raise CustomException(error_message=e, error_detail=sys)

