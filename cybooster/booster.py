from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from ._boosterc import BoosterRegressor, BoosterClassifier

class SkBoosterRegressor(BoosterRegressor, BaseEstimator, RegressorMixin):
    """A scikit-learn compatible wrapper for BoosterRegressor."""
    def fit(self, X, y):
        X = X.astype('float64')
        y = y.astype('float64')
        super().fit(X, y)
        return self

    def predict(self, X):
        X = X.astype('float64')
        return super().predict(X)

class SkBoosterClassifier(BoosterClassifier, BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible wrapper for BoosterClassifier."""
    def fit(self, X, y):
        X = X.astype('float64')
        y = y.astype('int64')
        super().fit(X, y)
        return self

    def predict(self, X):
        X = X.astype('float64')
        return super().predict(X)

    def predict_proba(self, X):
        X = X.astype('float64')
        return