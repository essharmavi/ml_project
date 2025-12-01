import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
# from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)

from src.logger import logging
from src.exception import CustomException
from src.components.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    model_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            # logging.info(f"Train array shape: {train_array.shape}")
            # logging.info(f"Test array shape: {test_array.shape}")        
            X_train, Y_train, X_test, Y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # logging.info("Shapes of data split ", X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                # "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "KNeighbors Regressor": KNeighborsRegressor(),
            }

            models_report: dict = evaluate_model(
                X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, models=models
            )

            logging.info(f"Models report: {models_report}")

            best_r2_score = max(sorted(models_report.values()))
            best_model_name = list(models_report.keys())[
                list(models_report.values()).index(best_r2_score)
            ]

            best_model = models[best_model_name]

            if best_r2_score < 0.6:
                raise CustomException("No good model found")

            logging.info(f"Best model - {best_model_name}")

            save_object(file_path=self.model_trainer_config.model_path, obj=best_model)

            predicted_output = best_model.predict(X_test)
            r2_score_get = r2_score(Y_test, predicted_output)
            return r2_score_get

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)
