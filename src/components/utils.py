import os
import sys
import dill
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

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
        hyperparameter_path = os.path.join(os.getcwd(),"config","hyperparameter_config.yaml")

        hyperparameters = load_config(hyperparameter_path)

        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            params = hyperparameters.get(model_name, {})

            grid_search = GridSearchCV(model, params, cv=3)
            grid_search.fit(X_train, Y_train)

            best_params = grid_search.best_params_

            model.set_params(**best_params)

            model.fit(X_train, Y_train)

            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)

            r2_score_train = r2_score(Y_train, Y_train_pred)
            r2_score_test = r2_score(Y_test, Y_test_pred)

            models_report[model_name] = r2_score_test
        
        return models_report

    except Exception as e:
        raise CustomException(error_message=e, error_detail=sys)
    
def load_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        raise CustomException(error_message=e, error_detail=sys)


