import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Dataset read into dataframe")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train Test Split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)


if __name__ == "__main__":
    ingestion_obj = DataIngestion()
    transformation_obj = DataTransformation()
    model_trainer_obj = ModelTrainer()

    train_path, test_path = ingestion_obj.initiate_data_ingestion()

    train_df_array,test_df_array = transformation_obj.initiate_data_transformation(train_path, test_path)

    get_score = model_trainer_obj.initiate_model_trainer(train_array = train_df_array,test_array=test_df_array)
    print("Best model's R2 score: ", get_score)



