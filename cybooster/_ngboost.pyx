# ============================================================================
# ngboost_cy.pyx - Corrected Core Cython Implementation
# ============================================================================

# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import nnetsauce as ns 
import numpy as np
cimport numpy as cnp
cimport cython
from copy import deepcopy
from libc.math cimport exp, log, sqrt, fabs
from cython.parallel import prange
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

ctypedef cnp.float64_t DTYPE_t

cdef double EPS = 1e-8  # Small constant for numerical stability

cdef class NGBRegressor:
    """Optimized NGBoost implementation"""
    
    cdef:
        cdef object obj
        cdef object fit_obj
        int n_estimators, n_samples, n_features
        double learning_rate, tol
        list learners, scalers
        cnp.ndarray params_
        cnp.ndarray natural_grads_  # Pre-allocated gradient array
        double base_mu, base_log_sigma
        bint is_fitted
        bint early_stopping
        int n_iter_no_change
        int verbose
        int feature_engineering
        
    def __init__(self, object obj=None, int n_estimators=500, double learning_rate=0.01, 
                 double tol=1e-4, bint early_stopping=True, int n_iter_no_change=10, int verbose=1, 
                 feature_engineering=0):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if n_iter_no_change <= 0:
            raise ValueError("n_iter_no_change must be positive")     
        self.obj = obj                
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.tol = tol
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose
        self.learners = []
        self.scalers = []
        self.is_fitted = False
        self.feature_engineering = feature_engineering  

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void compute_natural_gradients(self, cnp.ndarray[DTYPE_t, ndim=2] params, 
                                        cnp.ndarray[DTYPE_t, ndim=1] y,
                                        cnp.ndarray[DTYPE_t, ndim=2] grads):
        """Compute NATURAL gradients for Normal distribution"""
        cdef int n = params.shape[0]
        
        # Extract column vectors
        cdef cnp.ndarray[DTYPE_t, ndim=1] mu = params[:, 0]
        cdef cnp.ndarray[DTYPE_t, ndim=1] log_sigma = params[:, 1]
        
        # Vectorized operations
        cdef cnp.ndarray[DTYPE_t, ndim=1] sigma = np.exp(log_sigma)
        cdef cnp.ndarray[DTYPE_t, ndim=1] sigma_sq = np.maximum(sigma * sigma, EPS)
        cdef cnp.ndarray[DTYPE_t, ndim=1] diff = y - mu
        
        # NATURAL GRADIENTS: Fisher^{-1} @ score
        grads[:, 0] = diff  # natural_grad_μ = (y-μ)
        grads[:, 1] = 0.5 * (-1.0 + diff*diff / sigma_sq)  # natural_grad_logσ


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double compute_log_likelihood(self, cnp.ndarray[DTYPE_t, ndim=2] params,
                                    cnp.ndarray[DTYPE_t, ndim=1] y):
        """Compute log-likelihood for monitoring convergence"""
        cdef int n = params.shape[0]
        cdef double log_2pi = 1.8378770664093453  # log(2π)

        # Extract column vectors using NumPy slicing
        cdef cnp.ndarray[DTYPE_t, ndim=1] mu = params[:, 0]
        cdef cnp.ndarray[DTYPE_t, ndim=1] log_sigma = params[:, 1]

        # Vectorized operations
        cdef cnp.ndarray[DTYPE_t, ndim=1] sigma = np.maximum(np.exp(log_sigma), EPS)
        cdef cnp.ndarray[DTYPE_t, ndim=1] diff = y - mu

        # Compute log-likelihood terms in a vectorized manner
        cdef cnp.ndarray[DTYPE_t, ndim=1] ll_terms = -0.5 * (
            log_2pi + 2 * log_sigma + (diff * diff) / (sigma * sigma)
        )

        # Sum all terms
        return np.sum(ll_terms)
    
    @cython.boundscheck(False)
    cdef double compute_optimal_step_size(self, cnp.ndarray[DTYPE_t, ndim=1] gradients,
                                          cnp.ndarray[DTYPE_t, ndim=1] predictions):
        """Compute optimal step size using line search"""
        cdef double numerator = 0.0
        cdef double denominator = 0.0
        cdef int i, n = gradients.shape[0]
        
        for i in range(n):
            numerator += gradients[i] * predictions[i]
            denominator += predictions[i] * predictions[i]
        
        if fabs(denominator) < EPS:
            return 1.0
        
        return numerator / denominator
    
    def fit(self, cnp.ndarray[DTYPE_t, ndim=2] X, cnp.ndarray[DTYPE_t, ndim=1] y):
        """Fit NGBoost model with improved numerical stability"""
        self.n_samples, self.n_features = X.shape[0], X.shape[1]
        
        # Initialize parameters with numerical stability
        cdef cnp.ndarray[DTYPE_t, ndim=2] params = np.zeros((self.n_samples, 2), dtype=np.float64)
        cdef double y_mean = np.mean(y)
        cdef double y_std = np.std(y)
        
        # Ensure std is not too small
        if y_std < EPS:
            y_std = 1.0
            
        params[:, 0] = y_mean  # mu
        params[:, 1] = log(y_std)  # log_sigma
        
        # Store base predictions
        self.base_mu = y_mean
        self.base_log_sigma = log(y_std)
        
        # Pre-allocate gradient array
        self.natural_grads_ = np.zeros((self.n_samples, 2), dtype=np.float64)
        
        self.learners = []
        self.scalers = []
        
        cdef int iteration, no_improve_count = 0
        cdef double prev_ll = -np.inf, current_ll
        cdef cnp.ndarray[DTYPE_t, ndim=1] predictions
        cdef double scaler
        
        iterator = tqdm(range(self.n_estimators)) if self.verbose else range(self.n_estimators)
        for iteration in iterator:
            # Compute natural gradients (in-place)
            self.compute_natural_gradients(params, y, self.natural_grads_)
            
            # Compute current log-likelihood for monitoring
            current_ll = self.compute_log_likelihood(params, y)
            
            # Check for convergence
            if self.early_stopping and iteration > 0:
                if fabs(current_ll - prev_ll) < self.tol:
                    no_improve_count += 1
                    if no_improve_count >= self.n_iter_no_change:
                        print(f"Early stopping at iteration {iteration}")
                        break
                else:
                    no_improve_count = 0
            
            prev_ll = current_ll
            
            # Fit base learners for each parameter
            iter_learners = []
            iter_scalers = []
            
            for param_idx in range(2):
                # Fit base learner
                if self.obj is None: 
                    if self.feature_engineering:
                        learner = ns.CustomRegressor(DecisionTreeRegressor(
                            max_depth=3, 
                            random_state=42 + iteration,  # Different seed each iteration
                            min_samples_leaf=max(1, self.n_samples // 100)  # Adaptive min samples
                        ))
                    else: 
                        learner = DecisionTreeRegressor(
                            max_depth=3, 
                            random_state=42 + iteration,  # Different seed each iteration
                            min_samples_leaf=max(1, self.n_samples // 100)  # Adaptive min samples
                        )
                else: 
                    if self.feature_engineering:
                        learner = deepcopy(ns.CustomRegressor(self.obj))
                    else:
                        learner = deepcopy(self.obj)
                    try: 
                        learner.set_params(random_state=42 + iteration)
                    except Exception as e: 
                        try: 
                            learner.set_params(seed=42 + iteration)
                        except Exception as e: 
                            pass 
                learner.fit(X, self.natural_grads_[:, param_idx])
                
                # Get predictions
                predictions = learner.predict(X)
                
                # Compute optimal step size with numerical stability
                scaler = self.compute_optimal_step_size(
                    self.natural_grads_[:, param_idx], predictions
                )
                
                # Clip scaler to reasonable bounds
                scaler = max(-10.0, min(10.0, scaler))
                
                iter_learners.append(learner)
                iter_scalers.append(scaler)
                
                # Update parameters
                for i in range(self.n_samples):
                    params[i, param_idx] += self.learning_rate * scaler * predictions[i]
            
            self.learners.append(iter_learners)
            self.scalers.append(iter_scalers)
        
        self.params_ = params
        self.is_fitted = True
        return self
    
    def predict(self, cnp.ndarray[DTYPE_t, ndim=2] X, bint return_std=False):
        """Predict distribution parameters or point estimates"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        cdef int n_samples = X.shape[0]
        cdef cnp.ndarray[DTYPE_t, ndim=2] predictions = np.zeros((n_samples, 2), dtype=np.float64)
        cdef cnp.ndarray[DTYPE_t, ndim=2] result = np.zeros((n_samples, 2), dtype=np.float64)
        
        # Initialize with base predictions
        predictions[:, 0] = self.base_mu
        predictions[:, 1] = self.base_log_sigma
        
        # Add boosting contributions
        cdef int iteration, param_idx, i
        cdef cnp.ndarray[DTYPE_t, ndim=1] base_pred

        iterator = tqdm(range(len(self.learners))) if self.verbose else range(len(self.learners))
        
        for iteration in iterator: 
            for param_idx in range(2):
                base_pred = self.learners[iteration][param_idx].predict(X)
                for i in range(n_samples):
                    predictions[i, param_idx] += (self.learning_rate * 
                                                 self.scalers[iteration][param_idx] * 
                                                 base_pred[i])
        
        if return_std:
            # Return mu and sigma (not log_sigma)
            for i in range(n_samples):
                result[i, 0] = predictions[i, 0]  # mu
                result[i, 1] = exp(min(predictions[i, 1], 20.0))  # sigma = exp(log_sigma)
            return result[:, 0], result[:, 1]
        else:
            # Return just the mean predictions
            return predictions[:, 0]
    
    def predict_dist(self, cnp.ndarray[DTYPE_t, ndim=2] X):
        """Predict full distribution parameters (mu, sigma)"""
        return self.predict(X, return_std=True)
    
    def sample(self, cnp.ndarray[DTYPE_t, ndim=2] X, int n_samples=1):
        """Sample from the predicted distributions"""
        cdef cnp.ndarray[DTYPE_t, ndim=2] dist_params = self.predict_dist(X)
        cdef int n_points = X.shape[0]
        
        if n_samples == 1:
            # Single sample per point
            samples = np.random.normal(
                loc=dist_params[:, 0],
                scale=dist_params[:, 1]
            )
            return samples
        else:
            # Multiple samples per point
            samples = np.zeros((n_points, n_samples), dtype=np.float64)
            for i in range(n_points):
                samples[i, :] = np.random.normal(
                    loc=dist_params[i, 0],
                    scale=dist_params[i, 1],
                    size=n_samples
                )
            return samples
    
    def score(self, cnp.ndarray[DTYPE_t, ndim=2] X, cnp.ndarray[DTYPE_t, ndim=1] y):
        """Compute log-likelihood score"""
        cdef cnp.ndarray[DTYPE_t, ndim=2] predictions = self.predict_dist(X)
        cdef int n = X.shape[0]
        cdef double ll = 0.0
        cdef double log_2pi = 1.8378770664093453
        
        for i in range(n):
            mu = predictions[i, 0]
            sigma = predictions[i, 1]
            diff = y[i] - mu
            ll -= 0.5 * (log_2pi + 2*log(sigma) + diff*diff/(sigma*sigma))
        
        return ll / n  # Average log-likelihood
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility"""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'tol': self.tol,
            'early_stopping': self.early_stopping,
            'n_iter_no_change': self.n_iter_no_change
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            setattr(self, key, value)
        return self