# ============================================================================
# ngboost_classifier_cy.pyx - NGBoost Classifier with Multinomial Distribution
# ============================================================================

# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import nnetsauce as ns 
import numpy as np
cimport numpy as cnp
cimport cython
from copy import deepcopy
from libc.math cimport exp, log, fabs
from cython.parallel import prange
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

ctypedef cnp.float64_t DTYPE_t   # for X (features)
ctypedef cnp.int64_t ITYPE_t     # for y (labels)

cdef double EPS = 1e-8  # Small constant for numerical stability

cdef class NGBClassifier:
    """Optimized NGBoost Classifier with Multinomial distribution (softmax)"""
    
    cdef:
        cdef object obj
        cdef object fit_obj
        int n_estimators, n_samples, n_features, n_classes
        double learning_rate, tol
        list learners, scalers
        cnp.ndarray params_
        cnp.ndarray natural_grads_  # Pre-allocated gradient array
        cnp.ndarray base_logits
        cnp.ndarray classes_
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
        self.n_classes = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void softmax(self, cnp.ndarray[DTYPE_t, ndim=1] logits, 
                      cnp.ndarray[DTYPE_t, ndim=1] probs):
        """Numerically stable softmax computation"""
        cdef int k = logits.shape[0]
        cdef double max_logit = logits[0]
        cdef double sum_exp = 0.0
        cdef int i
        
        # Find max for numerical stability
        for i in range(1, k):
            if logits[i] > max_logit:
                max_logit = logits[i]
        
        # Compute exp and sum
        for i in range(k):
            probs[i] = exp(logits[i] - max_logit)
            sum_exp += probs[i]
        
        # Normalize
        for i in range(k):
            probs[i] /= sum_exp

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void compute_natural_gradients(self, cnp.ndarray[DTYPE_t, ndim=2] logits, 
                                        cnp.ndarray[DTYPE_t, ndim=2] y_onehot,
                                        cnp.ndarray[DTYPE_t, ndim=2] grads):
        """Compute NATURAL gradients for Multinomial/Categorical distribution
        
        For categorical with softmax: natural gradient = y - p (residual)
        where y is one-hot encoded and p is the softmax probabilities
        """
        cdef int n = logits.shape[0]
        cdef int k = logits.shape[1]
        cdef int i, j
        cdef cnp.ndarray[DTYPE_t, ndim=1] probs = np.zeros(k, dtype=np.float64)
        
        for i in range(n):
            # Compute softmax probabilities
            self.softmax(logits[i, :], probs)
            
            # Natural gradient: y - p
            for j in range(k):
                grads[i, j] = y_onehot[i, j] - probs[j]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double compute_log_likelihood(self, cnp.ndarray[DTYPE_t, ndim=2] logits,
                                       cnp.ndarray[DTYPE_t, ndim=2] y_onehot):
        """Compute categorical cross-entropy log-likelihood
        
        LL = sum_i sum_k y_ik * log(p_ik)
        """
        cdef int n = logits.shape[0]
        cdef int k = logits.shape[1]
        cdef double ll = 0.0
        cdef int i, j
        cdef cnp.ndarray[DTYPE_t, ndim=1] probs = np.zeros(k, dtype=np.float64)
        cdef double max_logit, sum_exp, log_sum_exp
        
        for i in range(n):
            # Log-sum-exp for numerical stability
            max_logit = logits[i, 0]
            for j in range(1, k):
                if logits[i, j] > max_logit:
                    max_logit = logits[i, j]
            
            sum_exp = 0.0
            for j in range(k):
                sum_exp += exp(logits[i, j] - max_logit)
            
            log_sum_exp = max_logit + log(sum_exp)
            
            # Compute log-likelihood for this sample
            for j in range(k):
                if y_onehot[i, j] > 0:
                    ll += logits[i, j] - log_sum_exp
        
        return ll
    
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
    
    def fit(self, cnp.ndarray[DTYPE_t, ndim=2] X, cnp.ndarray[ITYPE_t, ndim=1] y):
        """Fit NGBoost classifier with multinomial distribution"""
        self.n_samples, self.n_features = X.shape[0], X.shape[1]
        
        # Get unique classes
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        if self.n_classes < 2:
            raise ValueError("Need at least 2 classes for classification")
        
        # Create one-hot encoding
        cdef cnp.ndarray[DTYPE_t, ndim=2] y_onehot = np.zeros((self.n_samples, self.n_classes), 
                                                               dtype=np.float64)
        cdef dict class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        cdef int i
        
        for i in range(self.n_samples):
            y_onehot[i, class_to_idx[y[i]]] = 1.0
        
        # Initialize logits with class priors (log of class frequencies)
        cdef cnp.ndarray[DTYPE_t, ndim=2] logits = np.zeros((self.n_samples, self.n_classes), 
                                                             dtype=np.float64)
        cdef cnp.ndarray[DTYPE_t, ndim=1] class_counts = np.sum(y_onehot, axis=0)
        cdef cnp.ndarray[DTYPE_t, ndim=1] init_logits = np.zeros(self.n_classes, dtype=np.float64)
        
        # Compute log-priors (with smoothing)
        for i in range(self.n_classes):
            init_logits[i] = log((class_counts[i] + 1.0) / (self.n_samples + self.n_classes))
        
        # Initialize all samples with the same prior logits
        for i in range(self.n_samples):
            logits[i, :] = init_logits
        
        # Store base predictions
        self.base_logits = init_logits.copy()
        
        # Pre-allocate gradient array
        self.natural_grads_ = np.zeros((self.n_samples, self.n_classes), dtype=np.float64)
        
        self.learners = []
        self.scalers = []
        
        cdef int iteration, no_improve_count = 0
        cdef double prev_ll = -np.inf, current_ll
        cdef cnp.ndarray[DTYPE_t, ndim=1] predictions
        cdef double scaler
        cdef int class_idx
        
        iterator = tqdm(range(self.n_estimators)) if self.verbose else range(self.n_estimators)
        for iteration in iterator:
            # Compute natural gradients (in-place)
            self.compute_natural_gradients(logits, y_onehot, self.natural_grads_)
            
            # Compute current log-likelihood for monitoring
            current_ll = self.compute_log_likelihood(logits, y_onehot)
            
            # Check for convergence
            if self.early_stopping and iteration > 0:
                if fabs(current_ll - prev_ll) < self.tol:
                    no_improve_count += 1
                    if no_improve_count >= self.n_iter_no_change:
                        if self.verbose:
                            print(f"\nEarly stopping at iteration {iteration}")
                        break
                else:
                    no_improve_count = 0
            
            prev_ll = current_ll
            
            # Fit base learners for each class
            iter_learners = []
            iter_scalers = []
            
            for class_idx in range(self.n_classes):
                # Fit base learner
                if self.obj is None: 
                    if self.feature_engineering:
                        learner = ns.CustomRegressor(DecisionTreeRegressor(
                            max_depth=3, 
                            random_state=42 + iteration * self.n_classes + class_idx,
                            min_samples_leaf=max(1, self.n_samples // 100)
                        ))
                    else: 
                        learner = DecisionTreeRegressor(
                            max_depth=3, 
                            random_state=42 + iteration * self.n_classes + class_idx,
                            min_samples_leaf=max(1, self.n_samples // 100)
                        )
                else: 
                    if self.feature_engineering:
                        learner = deepcopy(ns.CustomRegressor(self.obj))
                    else:
                        learner = deepcopy(self.obj)
                    try: 
                        learner.set_params(random_state=42 + iteration * self.n_classes + class_idx)
                    except Exception as e: 
                        try: 
                            learner.set_params(seed=42 + iteration * self.n_classes + class_idx)
                        except Exception as e: 
                            pass
                
                learner.fit(X, self.natural_grads_[:, class_idx])
                
                # Get predictions
                predictions = learner.predict(X)
                
                # Compute optimal step size with numerical stability
                scaler = self.compute_optimal_step_size(
                    self.natural_grads_[:, class_idx], predictions
                )
                
                # Clip scaler to reasonable bounds
                scaler = max(-10.0, min(10.0, scaler))
                
                iter_learners.append(learner)
                iter_scalers.append(scaler)
                
                # Update logits
                for i in range(self.n_samples):
                    logits[i, class_idx] += self.learning_rate * scaler * predictions[i]
            
            self.learners.append(iter_learners)
            self.scalers.append(iter_scalers)
        
        self.params_ = logits
        self.is_fitted = True
        return self
    
    def predict_proba(self, cnp.ndarray[DTYPE_t, ndim=2] X):
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        cdef int i = 0    
        cdef int n_samples = X.shape[0]
        cdef cnp.ndarray[DTYPE_t, ndim=2] logits = np.zeros((n_samples, self.n_classes), 
                                                             dtype=np.float64)
        cdef cnp.ndarray[DTYPE_t, ndim=2] proba = np.zeros((n_samples, self.n_classes), 
                                                           dtype=np.float64)
        
        # Initialize with base predictions
        for i in range(n_samples):
            logits[i, :] = self.base_logits
        
        # Add boosting contributions
        cdef int iteration, class_idx
        cdef cnp.ndarray[DTYPE_t, ndim=1] base_pred

        iterator = tqdm(range(len(self.learners))) if self.verbose else range(len(self.learners))
        
        for iteration in iterator: 
            for class_idx in range(self.n_classes):
                base_pred = self.learners[iteration][class_idx].predict(X)
                for i in range(n_samples):
                    logits[i, class_idx] += (self.learning_rate * 
                                            self.scalers[iteration][class_idx] * 
                                            base_pred[i])
        
        # Convert logits to probabilities using softmax
        cdef cnp.ndarray[DTYPE_t, ndim=1] sample_probs = np.zeros(self.n_classes, dtype=np.float64)
        for i in range(n_samples):
            self.softmax(logits[i, :], sample_probs)
            proba[i, :] = sample_probs
        
        return proba
    
    def predict(self, cnp.ndarray[DTYPE_t, ndim=2] X):
        """Predict class labels"""
        cdef cnp.ndarray[DTYPE_t, ndim=2] proba = self.predict_proba(X)
        cdef cnp.ndarray[cnp.int64_t, ndim=1] class_indices = np.argmax(proba, axis=1)
        return self.classes_[class_indices]
    
    def predict_logit(self, cnp.ndarray[DTYPE_t, ndim=2] X):
        """Predict raw logit values"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        cdef int i = 0    
        cdef int n_samples = X.shape[0]
        cdef cnp.ndarray[DTYPE_t, ndim=2] logits = np.zeros((n_samples, self.n_classes), 
                                                             dtype=np.float64)
        
        # Initialize with base predictions
        for i in range(n_samples):
            logits[i, :] = self.base_logits
        
        # Add boosting contributions
        cdef int iteration, class_idx
        cdef cnp.ndarray[DTYPE_t, ndim=1] base_pred

        iterator = tqdm(range(len(self.learners))) if self.verbose else range(len(self.learners))
        
        for iteration in iterator: 
            for class_idx in range(self.n_classes):
                base_pred = self.learners[iteration][class_idx].predict(X)
                for i in range(n_samples):
                    logits[i, class_idx] += (self.learning_rate * 
                                            self.scalers[iteration][class_idx] * 
                                            base_pred[i])
        
        return logits
    
    def sample(self, cnp.ndarray[DTYPE_t, ndim=2] X, int n_samples=1):
        """Sample from the predicted categorical distributions"""
        cdef cnp.ndarray[DTYPE_t, ndim=2] proba = self.predict_proba(X)
        cdef int n_points = X.shape[0]
        
        if n_samples == 1:
            # Single sample per point
            samples = np.zeros(n_points, dtype=np.float64)
            for i in range(n_points):
                samples[i] = self.classes_[np.random.choice(self.n_classes, p=proba[i, :])]
            return samples
        else:
            # Multiple samples per point
            samples = np.zeros((n_points, n_samples), dtype=np.float64)
            for i in range(n_points):
                sample_indices = np.random.choice(self.n_classes, size=n_samples, p=proba[i, :])
                samples[i, :] = self.classes_[sample_indices]
            return samples
    
    def score(self, cnp.ndarray[DTYPE_t, ndim=2] X, cnp.ndarray[ITYPE_t, ndim=1] y):
        """Compute average log-likelihood score"""
        # Create one-hot encoding
        cdef cnp.ndarray[DTYPE_t, ndim=2] y_onehot = np.zeros((X.shape[0], self.n_classes), 
                                                               dtype=np.float64)
        cdef dict class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        
        for i in range(X.shape[0]):
            y_onehot[i, class_to_idx[y[i]]] = 1.0
        
        cdef cnp.ndarray[DTYPE_t, ndim=2] logits = self.predict_logit(X)
        cdef double ll = self.compute_log_likelihood(logits, y_onehot)
        return ll / X.shape[0]  # Average log-likelihood
    
    def accuracy_score(self, cnp.ndarray[DTYPE_t, ndim=2] X, cnp.ndarray[ITYPE_t, ndim=1] y):
        """Compute classification accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility"""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'tol': self.tol,
            'early_stopping': self.early_stopping,
            'n_iter_no_change': self.n_iter_no_change,
            'verbose': self.verbose,
            'feature_engineering': self.feature_engineering
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            setattr(self, key, value)
        return self