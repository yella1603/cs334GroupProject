from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import pandas as pd

trainDF = pd.read_csv('/Users/yellaleoniediekmann/cs334/selTrainDF.csv')
testDF = pd.read_csv('/Users/yellaleoniediekmann/cs334/selTestDF.csv')

X_train = trainDF.iloc[:, :-1].drop('score', axis=1)
y_train = trainDF.iloc[:, -1]

X_test = testDF.iloc[:, :-1].drop('score', axis=1)
y_test = testDF.iloc[:, -1]

base_models = [
    ('gbr', GradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=2,
        min_samples_split=2,
        n_estimators=100,
        random_state=42
    )),
    ('cbr', CatBoostRegressor(
        depth=6,
        iterations=100,
        l2_leaf_reg=5,
        learning_rate=0.05,
        random_state=42,
        verbose=0
    )),
    ('xgb', XGBRegressor(
        learning_rate=0.05,
        max_depth=3,
        n_estimators=100,
        subsample=1.0,
        random_state=42
    ))
]

meta_model = GradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=2,
    min_samples_split=2,
    n_estimators=100,
    random_state=42
)

stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Value:", r2)
