import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.logger import logging
from src.exception import CustomException
from src.components.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:

            num_features = ["writing_score", "reading_score"]
            cat_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            logging.info(f"Categorical features {cat_features}")
            logging.info(f"Numerical features {num_features}")

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            logging.info("Numerical Columns standard scaling completed")
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            logging.info("Categorical Columns encoding ompleted")

            preprocessor = ColumnTransformer(
                [
                    ("num_transformer", num_pipeline, num_features),
                    ("cat_transformer", cat_pipeline, cat_features),
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Training and Test dataset successfully read")

            logging.info("Obtaining preprocessing object")
            preprocessor = self.get_data_transformer_object()

            target_col_name = "math_score"


            input_feature_train_df = train_df.drop([target_col_name], axis=1)
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop([target_col_name], axis=1)
            target_feature_test_df = test_df[target_col_name]

            logging.info("Applying preprocessing on training and test data")

            input_feature_train_array = preprocessor.fit_transform(
                input_feature_train_df
            )

            input_feature_test_array = preprocessor.transform(input_feature_test_df)

            train_df_array = np.c_[
                input_feature_train_array, np.array(target_feature_train_df)
            ]
            test_df_array = np.c_[
                input_feature_test_array, np.array(target_feature_test_df)
            ]

            logging.info("Preprocessing completed")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessor,
            )

            return (
                train_df_array,
                test_df_array,
                self.data_transformation_config.preprocessor_obj_path,
            )

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)
