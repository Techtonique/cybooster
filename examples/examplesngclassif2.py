import numpy as np
from cybooster import NGBClassifier
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import ExtraTreeRegressor
from time import time 


datasets = [load_breast_cancer(), load_iris(), load_wine(), load_digits()]

for i, dataset in enumerate(datasets): 
    print("dataset #", i+1)
    X, y = dataset.data, dataset.target
    X = X.astype(np.float64)
    y = y.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    regressor = NGBClassifier(verbose=True)
    start = time()
    regressor.fit(X_train, y_train)
    print(f"Elapsed: {time() - start} s")
    y_pred = regressor.predict(X_test)
    print(np.mean(y_test == y_pred))

    regressor = NGBClassifier(LinearRegression(), verbose=True)
    start = time()
    regressor.fit(X_train, y_train)
    print(f"Elapsed: {time() - start} s")
    y_pred = regressor.predict(X_test)
    print(np.mean(y_test == y_pred))

