"""
CyBooster - A high-performance gradient boosting implementation using Cython

This package provides:
- BoosterRegressor: For regression tasks
- BoosterClassifier: For classification tasks
"""

from ._boosterc import BoosterRegressor, BoosterClassifier
from ._ngboost import NGBRegressor
from .ngboost import SkNGBRegressor
from .booster import SkBoosterRegressor, SkBoosterClassifier

__all__ = ["BoosterRegressor", "BoosterClassifier", 
           "SkBoosterRegressor", "SkBoosterClassifier",
           "NGBRegressor", "SkNGBRegressor"]  # Explicit exports
__version__ = "0.7.0"  # Package version
