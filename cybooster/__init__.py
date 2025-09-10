"""
CyBooster - A high-performance gradient boosting implementation using Cython

This package provides:
- BoosterRegressor: For regression tasks
- BoosterClassifier: For classification tasks
"""

from ._boosterc import BoosterRegressor, BoosterClassifier
from .booster import SkBoosterRegressor, SkBoosterClassifier

__all__ = ["BoosterRegressor", "BoosterClassifier", 
           "SkBoosterRegressor", "SkBoosterClassifier"]  # Explicit exports
__version__ = "0.4.0"  # Package version
