from cybooster import BoosterClassifier, BoosterRegressor
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from time import time 


# Regression Example
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regressor = BoosterRegressor(obj=LinearRegression(), n_estimators=100, learning_rate=0.1,
                             n_hidden_features=10, verbose=1, seed=42)
start = time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Elapsed: {time() - start} s")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")
print("X_test.shape:", X_test.shape)
sensi = regressor.compute_sensitivities(X_test,  activation='relu')
print("sensi:", sensi)
print("sensi.shape:", sensi.shape)

regressor = BoosterRegressor(obj=Ridge(), n_estimators=100, learning_rate=0.1,
                             n_hidden_features=10, verbose=1, seed=42)
start = time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Elapsed: {time() - start} s")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")
print("X_test.shape:", X_test.shape)
sensi = regressor.compute_sensitivities(X_test,  activation='relu')
print("sensi:", sensi)
print("sensi.shape:", sensi.shape)

