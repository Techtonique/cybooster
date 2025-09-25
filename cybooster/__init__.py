"""
CyBooster - A high-performance gradient boosting implementation using Cython

This package provides:
- BoosterRegressor: For regression tasks
- BoosterClassifier: For classification tasks
"""

from ._boosterc import BoosterRegressor, BoosterClassifier
from ._ngboost import NGBoost
from .ngboost import SkNGBoost
from .booster import SkBoosterRegressor, SkBoosterClassifier

__all__ = ["BoosterRegressor", "BoosterClassifier", 
           "SkBoosterRegressor", "SkBoosterClassifier",
           "NGBoost", "SkNGBoost"]  # Explicit exports
__version__ = "0.5.0"  # Package version
