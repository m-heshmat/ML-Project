import os 
import sys
from dataclasses import dataclass


from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from src.exceptions import CustomException
from src.logger import logging


from src.utils import save_object,evaluate_model

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_train_config=ModelTrainerConfig()

    def initiate_data_trainer(self,train_array,test_array):
        
        try:
            logging.info("splitting training and test input data")
            X_train,y_train,X_test,y_test=(

                #These two are often used to separate input features from target values in a dataset
                train_array[:,:-1], #take all rows except the last column =
                train_array[:,-1], #take the last column only
                test_array[:,:-1],
                test_array[:,-1]
            )

            
            models={
                
                "Linear Regression":LinearRegression(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Random Forest Regressor":RandomForestRegressor(),
                "Gradient Boosting Regressor":GradientBoostingRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor(),
            }

            
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            if model_report is None or not model_report:
                raise CustomException("Model evaluation failed. No results returned from evaluate_model.")
            logging.info(f"model report:{model_report}")

            best_model_score = max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_train_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square
        
        
        
        
        
        except Exception as e:

            raise CustomException(e,sys)
