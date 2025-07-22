from cybooster import BoosterClassifier, BoosterRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression


# Regression Example
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regressor = BoosterRegressor(obj=LinearRegression(), n_estimators=100, learning_rate=0.1,
                             n_hidden_features=10, verbose=1, seed=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for regression: {rmse:.4f}")

# Classification Example
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = BoosterClassifier(obj=LinearRegression(), n_estimators=100, learning_rate=0.1,
                               n_hidden_features=10, verbose=1, seed=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for classification: {accuracy:.4f}")
