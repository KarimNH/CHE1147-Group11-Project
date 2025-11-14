import os
import numpy as np
import io
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, r2_score,
                             mean_absolute_error, mean_squared_error, max_error, mean_absolute_percentage_error)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def get_regression_metrics(model, X, y_true):
    """
    Get a dicionary with regression metrics,
    including MAE, MSE, Max Error, MAPE, R2 Score
    """
    y_predicted = model.predict(X)

    mae = mean_absolute_error(y_true, y_predicted)
    mse = mean_squared_error(y_true, y_predicted)
    maximum_error = max_error(y_true, y_predicted)
    mape = mean_absolute_percentage_error(y_true, y_predicted)
    r2 = r2_score(y_true, y_predicted)

    metrics_dict = {
        'mae': mae,
        'mse': mse,
        'max_error': maximum_error,
        'mape': mape,
        'r2': r2
    }

    return metrics_dict

def dummy_regressor_baseline(X_train, y_train):
    """
    Build Dummy Regressor Baseline Models.
    """

    dummyregressor_mean = DummyRegressor(strategy='mean')
    dummyregressor_median = DummyRegressor(strategy='median')
    dummyregressor_mean.fit(X_train, y_train)
    dummyregressor_median.fit(X_train, y_train)

    dummy_regressors = [
        ('mean', dummyregressor_mean),
        ('median', dummyregressor_median)
    ]

    return dummy_regressors

# Add the function for linear regression and krr here...
def get_default_krr(X_train, y_train):
    """
    Returns an untuned KernelRidge Regressor using an RBF Kernel with default parameters.
    """
    model = KernelRidge(kernel='rbf')
    model.fit(X_train, y_train)
    
    return model

def get_default_xgb(X_train, y_train, random_seed=10):
    """
    Returns an untuned XGBRegressor with default parameters.
    """
    model = XGBRegressor(random_state=random_seed)
    model.fit(X_train, y_train)
    
    return model

def get_default_rf(X_train, y_train, random_seed=10):
    """
    Returns an untuned RandomForestRegressor with default parameters.
    """
    model = RandomForestRegressor(random_state=random_seed)
    model.fit(X_train, y_train)
    
    return model

def reg_cv(model_type, search_type, param, cv, scorer, X_train, y_train, n_iter = 500, random_seed=10):
    """
    Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
    for regression models (XGBRegressor or RandomForestRegressor)
    """

    if model_type == 'xgb':
        model = XGBRegressor(random_state=random_seed)
    elif model_type == 'rf':
        model = RandomForestRegressor(random_state=random_seed)
    else:
        raise ValueError("model_type must be 'xgb' or 'rf'")

    if search_type == "gridsearch":
        search = GridSearchCV(
            estimator=model,
            param_grid=param,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            verbose=10
            )
        
    elif search_type == 'randomsearch':
        search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param,
        scoring=scorer,
        cv=cv,
        n_iter=n_iter,
        random_state=random_seed,
        n_jobs=-1,
        verbose=10
        )
    else:
        raise ValueError("search_type must be 'gridsearch' or 'randomsearch'")
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    return search, best_model
