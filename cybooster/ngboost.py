import numpy as np
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

class SkNGBoost(BaseEstimator, RegressorMixin):
    """NGBoost wrapper with optional JAX acceleration"""
    
    def __init__(self, n_estimators=500, learning_rate=0.01, tol=1e-4, 
                 use_jax=True, verbose=False):
        self.use_jax = use_jax and JAX_AVAILABLE
        self.verbose = verbose
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.tol = tol
        
        try:
            from ._ngboost import NGBoost
            self.ngb = NGBoost(n_estimators, learning_rate, tol)
        except ImportError:
            warnings.warn("Cython module not available, using fallback")
            self.ngb = self._create_fallback()
            
        if self.use_jax:
            self._setup_jax()
    
    def _setup_jax(self):
        """Setup JIT-compiled JAX functions"""
        @jit
        def fast_matmul(A, B):
            return jnp.dot(A, B)
        self._fast_matmul = fast_matmul
    
    def fit(self, X, y):
        """Fit the model"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if self.verbose:
            print(f"Fitting NGBoost with {X.shape[0]} samples, {X.shape[1]} features")
            print(f"JAX enabled: {self.use_jax}")
        
        return self.ngb.fit(X, y)
    
    def predict(self, X, return_std=False):
        """Predict distribution parameters"""
        X = np.asarray(X, dtype=np.float64)
        return self.ngb.predict(X, return_std)
    
    def predict_dist(self, X):
        """Predict probability distributions"""
        params = self.predict(X)
        from scipy.stats import norm
        
        distributions = []
        for i in range(params.shape[0]):
            mu = params[i, 0]
            sigma = np.exp(params[i, 1])
            distributions.append(norm(loc=mu, scale=sigma))
        
        return distributions
    
    def _create_fallback(self):
        """Simple fallback when Cython unavailable"""
        class SimpleFallback:
            def __init__(self, n_est, lr, tol):
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
        
        return SimpleFallback(self.n_estimators, self.learning_rate, self.tol)

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
