"""
CyBooster - A high-performance gradient boosting implementation using Cython

This package provides:
- BoosterRegressor: For regression tasks
- BoosterClassifier: For classification tasks
"""

from ._boosterc import BoosterRegressor, BoosterClassifier

__all__ = ['BoosterRegressor', 'BoosterClassifier']  # Explicit exports
__version__ = '0.1.1'  # Package version