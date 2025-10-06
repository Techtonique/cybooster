
# `CyBooster`: A Gradient Boosting Library

[![PyPI - License](https://img.shields.io/pypi/l/cybooster)](./LICENSE) [![Downloads](https://pepy.tech/badge/cybooster)](https://pepy.tech/project/cybooster) [![Documentation](https://img.shields.io/badge/documentation-is_here-green)](https://techtonique.github.io/cybooster/)

`CyBooster` is a high-performance generic gradient boosting (any based learner can be used) library designed for classification and regression tasks. It is built on Cython (that is, C) for speed and efficiency. This version will also be more GPU friendly, thanks to JAX, making it suitable for large datasets.

Each base learner is augmented with a randomized neural network (a generalization of [https://www.researchgate.net/publication/346059361_LSBoost_gradient_boosted_penalized_nonlinear_least_squares](https://www.researchgate.net/publication/346059361_LSBoost_gradient_boosted_penalized_nonlinear_least_squares) to any base learner), which allows the model to learn complex patterns in the data. The library supports both classification and regression tasks, making it versatile for various machine learning applications.

`CyBooster` is born from `mlsauce`~~, that might be difficult to install on some systems~~. 


## Installation

To install `CyBooster`, you can use `pip` or `uv` (faster):

```bash
pip install cybooster
```

or 

```bash
uv pip install cybooster
```

From GitHub:

```bash
pip install git+https://github.com/Techtonique/cybooster.git
```

## Usage

### 1 - Model-agnostic boosting

```python 
from cybooster import BoosterClassifier, BoosterRegressor
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression
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

# Classification Example
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = BoosterClassifier(obj=LinearRegression(), n_estimators=100, learning_rate=0.1,
                               n_hidden_features=10, verbose=1, seed=42)
start = time()
try: 
    classifier.fit(X_train, y_train)
except Exception as e: # this is for Windows users
    y_train = y_train.astype('int32')
    classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(f"Elapsed: {time() - start} s")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for classification: {accuracy:.4f}")

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = BoosterClassifier(obj=LinearRegression(), n_estimators=100, learning_rate=0.1,
                               n_hidden_features=10, verbose=1, seed=42)
start = time()
try:
    classifier.fit(X_train, y_train)
except Exception as e: # this is for Windows users
    y_train = y_train.astype('int32')
    classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(f"Elapsed: {time() - start} s")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for classification: {accuracy:.4f}")

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = BoosterClassifier(obj=LinearRegression(), n_estimators=100, learning_rate=0.1,
                               n_hidden_features=10, verbose=1, seed=42)
start = time()
try: 
    classifier.fit(X_train, y_train)
except Exception as e: # this is for Windows users
    y_train = y_train.astype('int32')
    classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(f"Elapsed: {time() - start} s")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for classification: {accuracy:.4f}")

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = BoosterClassifier(obj=LinearRegression(), n_estimators=100, learning_rate=0.1,
                               n_hidden_features=10, verbose=1, seed=42)
start = time()
try: 
    classifier.fit(X_train, y_train)
except Exception as e: # this is for Windows users
    y_train = y_train.astype('int32')
    classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(f"Elapsed: {time() - start} s")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for classification: {accuracy:.4f}")
```

### 2 - Model-agnostic NGBoostRegressor 

```python
import numpy as np
from cybooster import NGBoostRegressor, SkNGBoostRegressor
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

regressor = NGBoostRegressor()
start = time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Elapsed: {time() - start} s")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")
print("return_std:", regressor.predict(X_test, return_std=True))

regressor = SkNGBoostRegressor()
start = time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Elapsed: {time() - start} s")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")
print("return_std:", regressor.predict(X_test, return_std=True))

regressor = NGBoostRegressor(LinearRegression())
start = time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Elapsed: {time() - start} s")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")
print("return_std:", regressor.predict(X_test, return_std=True))

regressor = SkNGBoostRegressor(LinearRegression())
start = time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Elapsed: {time() - start} s")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")
print("return_std:", regressor.predict(X_test, return_std=True))

regressor = NGBoostRegressor(Ridge())
start = time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Elapsed: {time() - start} s")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")
print("return_std:", regressor.predict(X_test, return_std=True))

regressor = SkNGBoostRegressor(Ridge())
start = time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Elapsed: {time() - start} s")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")
print("return_std:", regressor.predict(X_test, return_std=True))

regressor = NGBoostRegressor(ExtraTreeRegressor())
start = time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Elapsed: {time() - start} s")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")
print("return_std:", regressor.predict(X_test, return_std=True))

regressor = SkNGBoostRegressor(ExtraTreeRegressor())
start = time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Elapsed: {time() - start} s")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")
print("return_std:", regressor.predict(X_test, return_std=True))
```
