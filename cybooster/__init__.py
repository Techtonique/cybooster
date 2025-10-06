"""
CyBooster - A high-performance gradient boosting implementation using Cython

This package provides:
- BoosterRegressor: For regression tasks
- BoosterClassifier: For classification tasks
"""

from ._boosterc import BoosterRegressor, BoosterClassifier
from ._ngboost import NGBRegressor
from ._ngboostclf import NGBClassifier
from .ngboost import SkNGBRegressor, SkNGBClassifier
from .booster import SkBoosterRegressor, SkBoosterClassifier

__all__ = ["BoosterRegressor", "BoosterClassifier", 
           "SkBoosterRegressor", "SkBoosterClassifier",
           "NGBRegressor", "NGBClassifier", 
           "SkNGBRegressor", "SkNGBClassifier"]  # Explicit exports
__version__ = "0.8.0"  # Package version
