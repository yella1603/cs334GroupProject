import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample

trainDF = pd.read_csv('/Users/yellaleoniediekmann/cs334/selTrainDF.csv')
testDF = pd.read_csv('/Users/yellaleoniediekmann/cs334/selTestDF.csv')

X_train = trainDF.iloc[:, :-1].drop('score', axis=1)
y_train = trainDF.iloc[:, -1]

X_test = testDF.iloc[:, :-1].drop('score', axis=1)
y_test = testDF.iloc[:, -1]

n_estimators = 100  
np.random.seed(42) 

predictions = np.zeros((X_test.shape[0], n_estimators))

for i in range(n_estimators):
    X_train_sample, y_train_sample = resample(X_train, y_train, replace=True)

    model = GradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=2,
        min_samples_split=2,
        n_estimators=100,
        random_state=np.random.randint(10000) 
    )
    model.fit(X_train_sample, y_train_sample)

    predictions[:, i] = model.predict(X_test)

y_pred = predictions.mean(axis=1)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Value:", r2)
