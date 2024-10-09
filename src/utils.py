import os
import sys 
import numpy as np 
import pandas as pd 
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

import dill 

from sklearn.exceptions import NotFittedError

from src.exceptions import CustomException

# Function to save objects (e.g., models, preprocessing pipelines) to a file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  # Create directories if they don't exist

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)  # Use dill to save the object
    except Exception as e:
        raise CustomException(e, sys)

# Function to evaluate multiple models with grid search for hyperparameter tuning
def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        
        # Loop through each model in the models dictionary
        for model_name, model in models.items():
            para = param.get(model_name, {})  # Get corresponding parameters or empty dict
            
            try:
                # Check if parameters are empty, skip GridSearchCV if true
                if para:
                    # Perform GridSearchCV with cross-validation (cv=3)
                    gs = GridSearchCV(model, para, cv=3, refit=True)
                    gs.fit(X_train, y_train)  # Grid search will fit the model and find the best parameters
                    best_model = gs.best_estimator_
                else:
                    # If no hyperparameters to tune, fit the model directly
                    model.fit(X_train, y_train)
                    best_model = model

                # Use the best estimator to predict
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                # Calculate R-squared scores for both train and test sets
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                # Log the test score in the report dictionary
                report[model_name] = test_model_score

            except NotFittedError as e:
                logging.error(f"Model {model_name} failed to fit during GridSearchCV: {e}")
                report[model_name] = None  # If model fails, set its score to None
            except Exception as e:
                logging.error(f"Error in model {model_name} with parameters {para}: {e}")
                report[model_name] = None  # If any other error occurs, log and set score to None

        return report
    
    except Exception as e:
        logging.error(f"Error in evaluating models: {e}")
        return None

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
