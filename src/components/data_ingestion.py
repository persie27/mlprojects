import os
import sys
from pathlib import Path
sys.path.append(str(Path('src').parent.parent)) 
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformer
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path = os.path.join("artifacts","test.csv")
    raw_data_path = os.path.join("artifacts","raw_data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def start_data_ingestion(self):
        logging.info("Starting Data Ingestion")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Data is written into Dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok= True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index= False, header= True) 
            
            logging.info("Train Test Split initiated")      
            
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=31)
            train_set.to_csv(self.ingestion_config.train_data_path, index= False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index= False, header = True)
            
            logging.info("Train Test Split completed")
            logging.info("Data Ingestion completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
            
if __name__=='__main__':
    dataInges = DataIngestion()
    train_data, test_data = dataInges.start_data_ingestion()
    
    dataTrans = DataTransformer()
    train_arr, test_arr, _ = dataTrans.initiate_data_transformation(train_data, test_data)

    modelTrain= ModelTrainer()
    print(modelTrain.initiate_model_trainer(train_arr,test_arr))