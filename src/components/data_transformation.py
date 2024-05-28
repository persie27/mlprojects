import sys
import os

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","data_transformer.pkl")
        
class DataTransformer:
    def __init__(self):
        self.data_transform_config =  DataTransformConfig()
        
    def get_data_transformed(self):
        logging.info("Starting Data Transformation")
        try:
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Numerical columns: {numerical_features}")
            logging.info("Numerical Transformation completed")
            
            logging.info(f"Categorical features: {categorical_features}")
            logging.info("Categorical Transformation completed")
            
            data_transform = ColumnTransformer(
                transformers=[
                    ("numeric_pipelines", numerical_pipeline, numerical_features),
                    ("cat_pipelines", categorical_pipeline, categorical_features)
                ]
            )
            
            return data_transform
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Reading train and test data")
            
            transformation_obj = self.get_data_transformed()
            
            target_column = "math_score"
            numerical_features = ['reading_score', 'writing_score']
            
            input_feature_train_df = train_df.drop(columns = [target_column], axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns = [target_column], axis=1)
            target_feature_test_df = test_df[target_column]
            
            logging.info(f"Applying transformation object on train and test data")
            input_feature_train_arr = transformation_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = transformation_obj.transform(input_feature_test_df)
            
            # logging.info(f"input_feature_test_arr: {input_feature_test_arr}")
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            # logging.info(f"input_feature_test_arr: {input_feature_test_arr}")
            
            logging.info("Saving the transformed object")
            
            save_object(
                file_path = self.data_transform_config.preprocessor_obj_file_path,
                obj = transformation_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transform_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)

