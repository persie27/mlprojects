import os
import sys

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting train and test data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info(f"Model trainer **** X_train {X_train[:1,:]}")
            models = {
                "Linear Regression": LinearRegression(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(),
                "RandomForest Regressor": RandomForestRegressor(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "Decision Tree": DecisionTreeRegressor()
            }

            model_report:dict = evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test= y_test, models = models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No model has a score of more 0.6")
            logging.info(f"Best model found for both training and test data {best_model}")

            save_object(
                file_path=self.model_trainer_config.trained_file_path,
                obj=best_model
            )

            prediction = best_model.predict(X_test)
            rsquare_score = r2_score(y_test,prediction)

            return rsquare_score
        except Exception as e:
            raise CustomException(e, sys)