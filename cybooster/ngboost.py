import numpy as np
import nnetsauce as ns 
from typing import Optional
import warnings
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

try:
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

class SkNGBRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible NGBoost regressor (Python wrapper).

    This is a thin wrapper around the high-performance Cython implementation
    exposed as ``cybooster._ngboost.NGBRegressor``. It mirrors a scikit-learn
    estimator interface while providing an optional JAX acceleration hook for
    internal linear algebra utilities.

    Parameters
    ----------
    obj : Any, optional
        Placeholder for future objectives (kept for backward compatibility).
        The current Cython backend expects this positional slot.
    n_estimators : int, default=500
        Number of boosting iterations.
    learning_rate : float, default=0.01
        Shrinkage applied to each boosting step.
    tol : float, default=1e-4
        Tolerance used for early stopping monitoring in the backend.
    early_stopping : bool, default=True
        Whether to enable early stopping based on log-likelihood improvement.
    n_iter_no_change : int, default=10
        Number of successive iterations with change < ``tol`` to trigger stop.
    feature_engineering : bool, default=False
        If True, enables feature engineering through nnetsauce.
    use_jax : bool, default=True
        If True and JAX is available, enables small JIT-compiled helpers.
    verbose : bool, default=False
        If True, prints fitting diagnostics.

    Notes
    -----
    - The underlying predictive distribution is Normal with parameters
      (mu, log_sigma). The backend learns both via natural gradients.
    - ``predict(X, return_std=False)`` returns point estimates. When
      ``return_std=True``, it returns a 2D array with columns ``(mu, sigma)``.
    - Use ``predict_dist(X)`` to directly obtain distribution parameters
      as ``(mu, sigma)``.
    """
    
    def __init__(self, obj=None, n_estimators=500, learning_rate=0.01, 
    tol=1e-4, early_stopping=True, n_iter_no_change=10, feature_engineering=False,
                 use_jax=True, verbose=False):
        """Initialize the NGBoost regressor wrapper.

        See class docstring for parameter details.
        """
        self.use_jax = use_jax and JAX_AVAILABLE
        self.verbose = verbose
        self.obj = obj 
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.tol = tol
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.feature_engineering = feature_engineering
       
        try:
            from ._ngboost import NGBRegressor
            self.ngb = NGBRegressor(self.obj, self.n_estimators, self.learning_rate, self.tol, 
            self.early_stopping, self.n_iter_no_change, int(self.verbose), 
            int(feature_engineering))
        except ImportError:
            warnings.warn("Cython module not available, using fallback")
            self.ngb = self._create_fallback()
            
        if self.use_jax:
            self._setup_jax()
    
    def _setup_jax(self):
        """Setup small JIT-compiled helpers when JAX is available."""
        @jit
        def fast_matmul(A, B):
            return jnp.dot(A, B)
        self._fast_matmul = fast_matmul
    
    def fit(self, X, y):
        """Fit the NGBoost model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : SkNGBRegressor
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if self.verbose:
            print(f"Fitting NGBoost with {X.shape[0]} samples, {X.shape[1]} features")
            print(f"JAX enabled: {self.use_jax}")
        
        return self.ngb.fit(X, y)
    
    def predict(self, X, return_std=False):
        """Predict values or distribution parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        return_std : bool, default=False
            - If False, returns point predictions (mu).
            - If True, returns an array of shape (n_samples, 2) with
              columns ``(mu, sigma)``.

        Returns
        -------
        ndarray
            Either ``(n_samples,)`` array of means or ``(n_samples, 2)``
            with ``(mu, sigma)`` per sample.
        """
        X = np.asarray(X, dtype=np.float64)
        return self.ngb.predict(X, return_std)
    
    def predict_dist(self, X):
        """Predict Normal distributions for each input sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        list[scipy.stats.rv_continuous]
            A list of Normal distributions parameterized by predicted
            ``mu`` and ``sigma`` for each input sample.
        """
        params = self.predict(X)
        from scipy.stats import norm
        
        distributions = []
        for i in range(params.shape[0]):
            mu = params[i, 0]
            sigma = np.exp(params[i, 1])
            distributions.append(norm(loc=mu, scale=sigma))
        
        return distributions
    
    def _create_fallback(self):
        """Create a minimal NumPy-only fallback estimator.

        Used when the Cython extension cannot be imported. It predicts a
        constant Normal distribution estimated from the training targets.
        """
        class SimpleFallback:
            def __init__(self, n_est, lr, tol, early_stopping, n_iter_no_change):
                self.n_est, self.lr, self.tol = n_est, lr, tol
                self.fitted = False
            
            def fit(self, X, y):
                self.mean_, self.std_ = np.mean(y), np.std(y)
                self.fitted = True
                return self
            
            def predict(self, X):
                if not self.fitted:
                    raise ValueError("Not fitted")
                n = X.shape[0]
                return np.column_stack([np.full(n, self.mean_), np.full(n, np.log(self.std_))])
        
        return SimpleFallback(self.n_estimators, self.learning_rate, self.tol, self.early_stopping, self.n_iter_no_change)

# ============================================================================
# Evaluation Utilities
# ============================================================================

def evaluate_predictions(y_true, pred_dists):
    """Comprehensive evaluation metrics"""
    from sklearn.metrics import mean_squared_error, r2_score
    
    y_pred = np.array([d.mean() for d in pred_dists])
    y_std = np.array([d.std() for d in pred_dists])
    
    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Probabilistic metrics
    nll = -np.mean([d.logpdf(y_true[i]) for i, d in enumerate(pred_dists)])
    
    # Coverage (95% prediction intervals)
    lower = np.array([d.ppf(0.025) for d in pred_dists])
    upper = np.array([d.ppf(0.975) for d in pred_dists])
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    
    return {
        'mse': mse,
        'r2': r2,
        'nll': nll,
        'coverage_95': coverage,
        'mean_uncertainty': np.mean(y_std)
    }

def plot_results(y_true, pred_dists, X_test=None):
    """Plot predictions with uncertainty"""
    try:
        import matplotlib.pyplot as plt
        
        y_pred = np.array([d.mean() for d in pred_dists])
        y_std = np.array([d.std() for d in pred_dists])
        
        plt.figure(figsize=(12, 5))
        
        # Predictions vs actual
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual')
        
        # Uncertainty visualization
        plt.subplot(1, 2, 2)
        idx = np.argsort(y_pred)[:100]  # Show subset for clarity
        plt.fill_between(range(len(idx)), 
                        (y_pred - 2*y_std)[idx], 
                        (y_pred + 2*y_std)[idx], 
                        alpha=0.3, label='95% PI')
        plt.plot(y_pred[idx], 'b-', label='Predicted')
        plt.scatter(range(len(idx)), y_true[idx], alpha=0.7, s=20, label='Actual')
        plt.xlabel('Sample Index')
        plt.ylabel('Values')
        plt.title('Predictions with Uncertainty')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")


class SkNGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, obj=None, n_estimators=500, learning_rate=0.01, 
                 tol=1e-4, early_stopping=True, n_iter_no_change=10, feature_engineering=False,
                 use_jax=True, verbose=False):
        # Initialize the base regressor, or modify based on your needs
        self.model = ns.SimpleMultitaskClassifier(obj=SkNGBRegressor(obj=obj, 
        n_estimators=n_estimators, learning_rate=learning_rate, 
        tol=tol, early_stopping=early_stopping, n_iter_no_change=n_iter_no_change, 
        feature_engineering=feature_engineering, use_jax=use_jax, verbose=verbose))
    
    def fit(self, X, y):
        # Implement fit method
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        # Implement predict method
        return self.model.predict(X)
    
    def predict_proba(self, X):
        # Implement predict method
        return self.model.predict_proba(X)