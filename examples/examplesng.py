import numpy as np
from cybooster import NGBoost, SkNGBoost
from sklearn.datasets import load_diabetes, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import ExtraTreeRegressor
from time import time 


X, y = fetch_openml("boston", version=1, as_frame=True, return_X_y=True)
cols = list(X.columns)
print("columns", cols)
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
X_train = np.asarray(X_train, dtype=np.float64)
y_train = np.asarray(y_train, dtype=np.float64)
X_test = np.asarray(X_test, dtype=np.float64)
y_test = np.asarray(y_test, dtype=np.float64)

regressor = NGBoost()
start = time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Elapsed: {time() - start} s")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")
print("return_std:", regressor.predict(X_test, return_std=True))

regressor = SkNGBoost()
start = time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Elapsed: {time() - start} s")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")
print("return_std:", regressor.predict(X_test, return_std=True))

regressor = NGBoost(LinearRegression())
start = time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Elapsed: {time() - start} s")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")
print("return_std:", regressor.predict(X_test, return_std=True))

regressor = SkNGBoost(LinearRegression())
start = time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Elapsed: {time() - start} s")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")
print("return_std:", regressor.predict(X_test, return_std=True))

regressor = NGBoost(Ridge())
start = time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Elapsed: {time() - start} s")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")
print("return_std:", regressor.predict(X_test, return_std=True))

regressor = SkNGBoost(Ridge())
start = time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Elapsed: {time() - start} s")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")
print("return_std:", regressor.predict(X_test, return_std=True))

regressor = NGBoost(ExtraTreeRegressor())
start = time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Elapsed: {time() - start} s")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")
print("return_std:", regressor.predict(X_test, return_std=True))

regressor = SkNGBoost(ExtraTreeRegressor())
start = time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Elapsed: {time() - start} s")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")
print("return_std:", regressor.predict(X_test, return_std=True))