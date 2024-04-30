import argparse
import json
import warnings
import pandas as pd
import time
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor

def eval_gridsearch(clf, pgrid, xTrain, yTrain, xTest, yTest):
    print("Starting Grid Search...")
    start = time.time()
    grid_search = GridSearchCV(estimator=clf, param_grid=pgrid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(xTrain, yTrain)
    timeElapsed = time.time() - start
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(xTest)
    mse = mean_squared_error(yTest, y_pred)
    r2 = r2_score(yTest, y_pred)
    residuals = yTest - y_pred
    
    results = {
        "MSE": mse,
        "R2": r2,
        "Time": timeElapsed
    }
    residual_data = {
        "residuals": residuals,
        "predicted": y_pred
    }
    optimal_params = grid_search.best_params_

    return results, residual_data, optimal_params

def eval_randomsearch(clf, pgrid, xTrain, yTrain, xTest, yTest):
    print("Starting Random Search...")
    total_combinations = 1
    for i in pgrid:
        total_combinations *= len(pgrid[i])
    final_iter = round(total_combinations / 3)
    if final_iter < 1:
        final_iter = 1
    start = time.time()
    random_search = RandomizedSearchCV(estimator=clf, param_distributions=pgrid, n_iter=final_iter, cv=5, scoring='neg_mean_squared_error')
    random_search.fit(xTrain, yTrain)
    timeElapsed = time.time() - start
    best_clf = random_search.best_estimator_
    y_pred = best_clf.predict(xTest)
    mse = mean_squared_error(yTest, y_pred)
    r2 = r2_score(yTest, y_pred)
    residuals = yTest - y_pred
    results = {
        "MSE": mse,
        "R2": r2,
        "Time": timeElapsed
    }
    residual_data = {
        "residuals": residuals,
        "predicted": y_pred
    }
    optimal_params = random_search.best_params_

    return results, residual_data, optimal_params

def eval_searchcv(modelName, model, paramGrid, xTrain, yTrain, xTest, yTest, perfDict, modelDataDF, bestParamDict):
    gridSearchResults, gridSearchData, gridSearchParams = eval_gridsearch(model, paramGrid, xTrain, yTrain, xTest, yTest)
    perfDict[modelName + " (Grid)"] = gridSearchResults
    
    gridSearchDataFrame = pd.DataFrame(gridSearchData)
    gridSearchDataFrame["model"] = modelName + " (Grid)"
    gridSearchDataFrame["actual"] = yTest.reset_index(drop=True)  
    modelDataDF = pd.concat([modelDataDF, gridSearchDataFrame], ignore_index=True)
    
    randomSearchResults, randomSearchData, randomSearchParams = eval_randomsearch(model, paramGrid, xTrain, yTrain, xTest, yTest)
    perfDict[modelName + " (Random)"] = randomSearchResults

    bestParamDict[modelName] = {"Grid": gridSearchParams, "Random": randomSearchParams}
    return perfDict, modelDataDF, bestParamDict


def get_parameter_grid(mName):
    """
    Given a model name, return the parameter grid associated with it

    Parameters
    ----------
    mName : string
        name of the model (e.g., DT, KNN, LR (None))

    Returns
    -------
    pGrid: dict
        A Python dictionary with the appropriate parameters for the model.
        The dictionary should have at least 2 keys and each key should have
        at least 2 values to try.
    """
    
    parameter_grid = {}

    if mName == 'GBM':
        parameter_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 4],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [1, 2]
        }
    elif mName == 'NN':
        parameter_grid = {
            "alpha": [0.01, 0.1, 1, 10],
            "max_iter": [1000, 2000]
        }
    elif mName == 'RFG':
        parameter_grid = {
            "n_estimators": [50, 100],
            "max_depth": [10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    elif mName == 'CBR':
        parameter_grid = {
            "iterations": [100, 300],
            "learning_rate": [0.01, 0.05],
            "depth": [4, 6],
            "l2_leaf_reg": [3, 5]
        }
    elif mName == 'XGB':
        parameter_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0]
        }
    elif mName == 'LGBM':
        parameter_grid = {
            'max_depth': [5, 10],
            'learning_rate': [0.1, 0.3],
            'n_estimators': [50, 100],
            'num_leaves': [50, 100]
        }
    return parameter_grid


def main():
    parser = argparse.ArgumentParser(description='Model Performance Evaluation and Parameter Tuning.')
    parser.add_argument("data_file", help="Filename for the dataset containing features and target values.")
    parser.add_argument("best_params_output", help="JSON filename to output the best parameters found.")
    parser.add_argument("residuals_output", help="CSV filename to output residual plots data.")
    args = parser.parse_args()

    dataset = pd.read_csv(args.data_file)
    features = dataset.iloc[:, :-1].drop('score', axis=1)
    target = dataset.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=42)
    
    performance_dict = {}
    best_parameters_dict = {}
    residuals_data_frame = pd.DataFrame()

    # Model Evaluation for Gradient Boosting Machine
    print("Tuning GBM (Gradient Boosting Machine) --------")
    model_name = "GBM"
    param_grid = get_parameter_grid(model_name)
    model = GradientBoostingRegressor()
    performance_dict, residuals_data_frame, best_parameters_dict = eval_searchcv(
        model_name, model, param_grid, x_train, y_train, x_test, y_test, performance_dict, residuals_data_frame, best_parameters_dict)
    
    # Model Evaluation for NN
    print("Tuning NN --------")
    model_name = "NN"
    param_grid = get_parameter_grid(model_name)
    model = MLPRegressor()
    performance_dict, residuals_data_frame, best_parameters_dict = eval_searchcv(
        model_name, model, param_grid, x_train, y_train, x_test, y_test, performance_dict, residuals_data_frame, best_parameters_dict)

    # Model Evaluation for Random Forest Regressor
    print("Tuning Random Forest Regressor --------")
    model_name = "RFG"
    param_grid = get_parameter_grid(model_name)
    model = RandomForestRegressor(n_estimators=100)
    performance_dict, residuals_data_frame, best_parameters_dict = eval_searchcv(
        model_name, model, param_grid, x_train, y_train, x_test, y_test, performance_dict, residuals_data_frame, best_parameters_dict)

    # Model Evaluation for CatBoostRegressor
    print("Tuning CatBoostRegressor --------")
    model_name = "CBR"
    param_grid = get_parameter_grid(model_name)
    model = CatBoostRegressor()
    performance_dict, residuals_data_frame, best_parameters_dict = eval_searchcv(
        model_name, model, param_grid, x_train, y_train, x_test, y_test, performance_dict, residuals_data_frame, best_parameters_dict)

    # Model Evaluation for XGBoost Regressor
    print("Tuning XGBoost Regressor --------")
    model_name = "XGB"
    param_grid = get_parameter_grid(model_name)
    model = XGBRegressor()
    performance_dict, residuals_data_frame, best_parameters_dict = eval_searchcv(
        model_name, model, param_grid, x_train, y_train, x_test, y_test, performance_dict, residuals_data_frame, best_parameters_dict)

    # Model Evaluation for LGBMRegressor
    print("Tuning LGBMRegressor --------")
    model_name = "LGBM"
    param_grid = get_parameter_grid(model_name)
    model = LGBMRegressor()
    performance_dict, residuals_data_frame, best_parameters_dict = eval_searchcv(
        model_name, model, param_grid, x_train, y_train, x_test, y_test, performance_dict, residuals_data_frame, best_parameters_dict)

    performance_dict = dict(sorted(performance_dict.items(), key=lambda item: item[1]['MSE']))

    for model, metrics in performance_dict.items():
        print(f"{model} - MSE: {metrics['MSE']:.4f}, R2: {metrics['R2']:.4f}")

    residuals_data_frame.to_csv(args.residuals_output, index=False)
    with open(args.best_params_output, 'w') as file:
        json.dump(best_parameters_dict, file)

if __name__ == "__main__":
    main()
