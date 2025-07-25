# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

import numpy as np
import jax.numpy as jnp
import sys 
cimport numpy as np
from jax import device_put
from copy import deepcopy
from tqdm import tqdm
from scipy.special import expit
from scipy import sparse

cdef class BoosterRegressor:
    cdef object obj
    cdef object fit_obj
    cdef int n_estimators
    cdef double learning_rate
    cdef int n_hidden_features
    cdef double reg_lambda
    cdef double alpha
    cdef double row_sample
    cdef double col_sample
    cdef double dropout
    cdef double tolerance
    cdef int direct_link
    cdef int verbose
    cdef int seed
    cdef str backend
    cdef str activation
    cdef str weights_distr
    
    def __init__(self, object obj, int n_estimators=100, double learning_rate=0.1, 
                 int n_hidden_features=5, double reg_lambda=0.1, 
                 double alpha=0.5, double row_sample=1, double col_sample=1,
                 double dropout=0, double tolerance=1e-6, int direct_link=1, 
                 int verbose=1, int seed=123, str backend="cpu", 
                 str activation="relu",
                 str weights_distr="uniform"):
        self.obj = obj
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_hidden_features = n_hidden_features
        self.reg_lambda = reg_lambda
        self.alpha = alpha
        self.row_sample = row_sample
        self.col_sample = col_sample
        self.dropout = dropout
        self.tolerance = tolerance
        self.direct_link = direct_link
        self.verbose = verbose
        self.seed = seed
        self.backend = backend
        self.activation = activation
        self.weights_distr = weights_distr
        self.fit_obj = None
        
    
    def fit(self, double[:,::1] X, double[:] y):
        self.fit_obj = fit_booster_regressor(
            X=X, y=y, n_estimators=self.n_estimators, 
            learning_rate=self.learning_rate, 
            n_hidden_features=self.n_hidden_features,
            reg_lambda=self.reg_lambda, alpha=self.alpha,
            row_sample=self.row_sample, col_sample=self.col_sample,
            dropout=self.dropout, tolerance=self.tolerance,
            direct_link=self.direct_link, verbose=self.verbose,
            seed=self.seed, backend=self.backend, 
            activation=self.activation, weights_distr=self.weights_distr,
            obj=self.obj
        )
        return self
    
    def predict(self, double[:,::1] X):
        if self.fit_obj is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return predict_booster_regressor(self.fit_obj, X, self.backend)
    
    def update(self, double[:] X, y, double alpha=0.5):
        if self.fit_obj is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        self.fit_obj = update_booster(self.fit_obj, X, y, alpha, self.backend)
        return self

cdef class BoosterClassifier:
    cdef object obj
    cdef object fit_obj
    cdef int n_estimators
    cdef double learning_rate
    cdef int n_hidden_features
    cdef double reg_lambda
    cdef double alpha
    cdef double row_sample
    cdef double col_sample
    cdef double dropout
    cdef double tolerance
    cdef int direct_link
    cdef int verbose
    cdef int seed
    cdef str backend
    cdef str activation
    cdef str weights_distr
    
    def __init__(self, object obj, int n_estimators=100, double learning_rate=0.1, 
                 int n_hidden_features=5, double reg_lambda=0.1, 
                 double alpha=0.5, double row_sample=1, double col_sample=1,
                 double dropout=0, double tolerance=1e-6, int direct_link=1, 
                 int verbose=1, int seed=123, str backend="cpu", 
                 str activation="relu",
                 str weights_distr="uniform"):
        self.obj = obj 
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_hidden_features = n_hidden_features
        self.reg_lambda = reg_lambda
        self.alpha = alpha
        self.row_sample = row_sample
        self.col_sample = col_sample
        self.dropout = dropout
        self.tolerance = tolerance
        self.direct_link = direct_link
        self.verbose = verbose
        self.seed = seed
        self.backend = backend
        self.activation = activation
        self.weights_distr = weights_distr
        self.fit_obj = None
    
    def fit(self, double[:,::1] X, long int[:] y, object obj=None):
        self.fit_obj = fit_booster_classifier(
            X=X, y=y, n_estimators=self.n_estimators, 
            learning_rate=self.learning_rate, 
            n_hidden_features=self.n_hidden_features,
            reg_lambda=self.reg_lambda, alpha=self.alpha,
            row_sample=self.row_sample, col_sample=self.col_sample,
            dropout=self.dropout, tolerance=self.tolerance,
            direct_link=self.direct_link, verbose=self.verbose,
            seed=self.seed, backend=self.backend, 
            activation=self.activation, weights_distr=self.weights_distr,
            obj=self.obj
        )
        return self
    
    def predict_proba(self, double[:,::1] X):
        if self.fit_obj is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return predict_proba_booster_classifier(self.fit_obj, X, self.backend)
    
    def predict(self, double[:,::1] X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def update(self, double[:] X, y, double alpha=0.5):
        if self.fit_obj is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        self.fit_obj = update_booster(self.fit_obj, X, y, alpha, self.backend)
        return self

# 0 - utils -----

# 0 - 1 data structures & funcs -----

# a tuple of doubles
cdef struct mydoubletuple:
    double elt1
    double elt2

DTYPE_double = np.double

DTYPE_int = np.int32

ctypedef np.double_t DTYPE_double_t


# 0 - 2 data structures & funcs -----

# dropout
def dropout_func(x, drop_prob=0, seed=123):

    assert 0 <= drop_prob <= 1

    n, p = x.shape

    if drop_prob == 0:
        return np.asarray(x, dtype=np.float64)

    if drop_prob == 1:
        return np.asarray(a=np.zeros_like(x), dtype=np.float64)

    np.random.seed(seed)
    dropped_indices = np.random.rand(n, p) > drop_prob

    return np.asarray(a=dropped_indices * x / (1 - drop_prob), dtype=np.float64)
    
# one-hot encoder for discrete response
def one_hot_encode(long int[:] y, 
                   int n_classes):
    
    cdef long int i 
    cdef long int n_obs = len(y)
    cdef double[:,::1] res = np.zeros((n_obs, n_classes), dtype=DTYPE_double)        

    for i in range(n_obs):
        res[i, y[i]] = 1

    return np.asarray(dtype=np.float64, a=res)

def one_hot_encode2(long int y, int n_classes):
  cdef double[:] res = np.zeros(n_classes)
  res[y] = 1
  return np.asarray(dtype=np.float64, a=res)
    
# 0 - 3 activation functions ----- 

def relu_activation(x):
    return np.maximum(x, 0)

def relu6_activation(x):
    return np.minimum(np.maximum(x, 0), 6)

def sigmoid_activation(x):
    return 1/(1 + np.exp(-x))

def activation_choice(x):
  activation_options = {
                "relu": relu_activation,
                "relu6": relu6_activation,
                "tanh": np.tanh,
                "sigmoid": sigmoid_activation
            }
  return activation_options[x]  


# 1 - classifier ----- 

# 1 - 1 fit classifier ----- 

def fit_booster_classifier(double[:,::1] X, long int[:] y, 
                           int n_estimators=100, double learning_rate=0.1, 
                           int n_hidden_features=5, double reg_lambda=0.1, 
                           double alpha=0.5, 
                           double row_sample=1, double col_sample=1,
                           double dropout=0, double tolerance=1e-6, 
                           int direct_link=1, int verbose=1,
                           int seed=123, str backend="cpu", 
                           str activation="relu",
                           str weights_distr = "uniform",
                           object obj = None
                           ): 
  
  cdef long int n
  cdef int p
  cdef int n_classes
  cdef Py_ssize_t iter
  cdef dict res
  cdef double ym
  cdef double[:] xm, xsd
  cdef double[:,::1] Y, X_, E, W_i, h_i, hh_i, hidden_layer_i, hhidden_layer_i
  cdef double current_error
  
  # Check dtype on Windows (optional but user-friendly)
  if sys.platform == 'win32' and np.asarray(y).dtype != np.int32:
      y = y.astype(np.int32)
  
  n = X.shape[0]
  p = X.shape[1]
  res = {}
  current_error = 1000000.0
  
  xm = np.asarray(dtype=np.float64, a=X).mean(axis=0)
  xsd = np.asarray(dtype=np.float64, a=X).std(axis=0)
  for i in range(len(xsd)):
    if xsd[i] == 0:
      xsd[i] = 1.0 

  
  res['direct_link'] = direct_link
  res['xm'] = np.asarray(dtype=np.float64, a=xm)
  res['xsd'] = np.asarray(dtype=np.float64, a=xsd)
  res['n_estimators'] = n_estimators
  res['learning_rate'] = learning_rate
  res['W_i'] = {}
  res['fit_obj_i'] = {} 
  res['col_index_i'] = {}
  res['loss'] = []
  res['activation'] = activation
  res['weights_distr'] = weights_distr
  
  X_ = (np.asarray(dtype=np.float64, a=X) - xm[None, :])/xsd[None, :]
  n_classes = len(np.unique(y))
  res['n_classes'] = n_classes
  res['n_obs'] = n 
  
  Y = one_hot_encode(y, n_classes)
  Ym = np.mean(Y, axis=0)
  res['Ym'] = Ym
  E = Y - Ym
  iterator = tqdm(range(n_estimators)) if verbose else range(n_estimators)

  fit_obj = obj 

  for iter in iterator:
      
      np.random.seed(seed + iter*1000)
    
      iy = np.sort(np.random.choice(a=range(p), 
                                    size=np.int32(p*col_sample), 
                                    replace=False), 
                   kind='quicksort')
      res['col_index_i'][iter] = iy                     
      X_iy = np.asarray(dtype=np.float64, a=X_)[:, iy] # must be X_!
      if res['weights_distr' ]== "uniform":
        W_i = np.random.rand(X_iy.shape[1], n_hidden_features)
      else: 
        W_i = np.random.randn(X_iy.shape[1], n_hidden_features)
      hhidden_layer_i = dropout_func(x=activation_choice(activation)(safe_sparse_dot(np.asarray(dtype=np.float64, a=X_iy), np.asarray(dtype=np.float64, a=W_i), backend)), 
                                     drop_prob=dropout, seed=seed)
      hh_i = np.hstack((X_iy, hhidden_layer_i)) if direct_link else hhidden_layer_i
      
      if row_sample < 1:
      
        ix = np.sort(np.random.choice(a=range(n), 
                                    size=np.int32(n*row_sample), 
                                    replace=False), 
                     kind='quicksort')
        X_iy_ix = X_iy[ix,:]       
        hidden_layer_i = dropout_func(x=activation_choice(activation)(safe_sparse_dot(X_iy_ix, W_i, backend)), 
                                      drop_prob=dropout, seed=seed)
        h_i =  np.hstack((X_iy_ix, hidden_layer_i)) if direct_link else hidden_layer_i
        fit_obj.fit(X = np.asarray(dtype=np.float64, a=h_i), y = np.asarray(dtype=np.float64, a=E)[ix,:])
                                 
      else:
      
        fit_obj.fit(X = np.asarray(dtype=np.float64, a=hh_i), y = np.asarray(dtype=np.float64, a=E))
            
      E = E - learning_rate*np.asarray(dtype=np.float64, a=fit_obj.predict(np.asarray(dtype=np.float64, a=hh_i)))
      
      res['W_i'][iter] = np.asarray(dtype=np.float64, a=W_i)
            
      res['fit_obj_i'][iter] = deepcopy(fit_obj)

      current_error = np.linalg.norm(E, ord='fro')

      res['loss'].append(current_error)
      
      try:
        if np.abs(np.flip(np.diff(res['loss'])))[0] <= tolerance:
          res['n_estimators'] = iter
          break
      except:
        pass
      
  return res
  
  
# 1 - 2 predict classifier ----- 

def predict_proba_booster_classifier(object obj, double[:,::1] X, str backend="cpu"):

  cdef int iter, n_estimators, n_classes
  cdef double learning_rate
  cdef double[:,::1] preds_sum, out_probs
  cdef long int n_row_preds 
  

  n_classes = obj['n_classes']
  direct_link = obj['direct_link']
  n_estimators = obj['n_estimators']
  learning_rate = obj['learning_rate']
  activation = obj['activation']
  X_ = (X - obj['xm'][None, :])/obj['xsd'][None, :]
  n_row_preds = X.shape[0]
  preds_sum = np.zeros((n_row_preds, n_classes))
  out_probs = np.zeros((n_row_preds, n_classes))
  
  
  for iter in range(n_estimators):
  
    iy = obj['col_index_i'][iter]
    X_iy = X_[:, iy] # must be X_!
    W_i = obj['W_i'][iter]
    hh_i = np.hstack((X_iy, activation_choice(activation)(safe_sparse_dot(np.asarray(dtype=np.float64, a=X_iy), np.asarray(dtype=np.float64, a=W_i), backend)))) if direct_link else activation_choice(activation)(safe_sparse_dot(np.asarray(dtype=np.float64, a=X_iy), np.asarray(dtype=np.float64, a=W_i), backend))
    # works because the regressor is Multitask 
    preds_sum = preds_sum + learning_rate*np.asarray(dtype=np.float64, a=obj['fit_obj_i'][iter].predict(np.asarray(dtype=np.float64, a=hh_i)))
  
  out_probs = expit(np.tile(obj['Ym'], n_row_preds).reshape(n_row_preds, n_classes) + np.asarray(dtype=np.float64, a=preds_sum))
  
  out_probs = out_probs/np.sum(out_probs, axis=1)[:, None]

  return np.asarray(dtype=np.float64, a=out_probs)
  
  
# 2 - regressor -----   

# 2 - 1 fit regressor -----
  
def fit_booster_regressor(double[:,::1] X, double[:] y, 
                           int n_estimators=100, double learning_rate=0.1, 
                           int n_hidden_features=5, double reg_lambda=0.1, 
                           double alpha=0.5, 
                           double row_sample=1, double col_sample=1,
                           double dropout=0, double tolerance=1e-6, 
                           int direct_link=1, int verbose=1, 
                           int seed=123, str backend="cpu", 
                           str activation="relu", 
                           str weights_distr = "uniform",
                           object obj = None): 
  
  cdef long int n
  cdef int i, p
  cdef int n_classes 
  cdef Py_ssize_t iter
  cdef dict res
  cdef double ym
  cdef double[:] xm, xsd, e
  cdef double[:,::1] X_, W_i, h_i, hh_i, hidden_layer_i, hhidden_layer_i
  cdef double current_error
  
  
  n = X.shape[0]
  p = X.shape[1]
  res = {}
  current_error = 1000000.0
  
  xm = np.asarray(dtype=np.float64, a=X).mean(axis=0)
  xsd = np.asarray(dtype=np.float64, a=X).std(axis=0)
  for i in range(len(xsd)):
    if xsd[i] == 0:
      xsd[i] = 1.0 

  
  res['direct_link'] = direct_link
  res['xm'] = np.asarray(dtype=np.float64, a=xm)
  res['xsd'] = np.asarray(dtype=np.float64, a=xsd)
  res['n_estimators'] = n_estimators
  res['learning_rate'] = learning_rate
  res['activation'] = activation
  res['W_i'] = {}
  res['fit_obj_i'] = {} 
  res['col_index_i'] = {}
  res['loss'] = []
  res['weights_distr'] = weights_distr
  res['n_obs'] = n 
  
  X_ = (np.asarray(dtype=np.float64, a=X) - xm[None, :])/xsd[None, :]
  n_classes = len(np.unique(y))
  res['n_classes'] = n_classes
  
  ym = np.mean(y)
  res['ym'] = ym
  e = y - np.repeat(ym, n)
  iterator = tqdm(range(n_estimators)) if verbose else range(n_estimators)

  fit_obj = obj 

  for iter in iterator:
      
      np.random.seed(seed + iter*1000)
    
      iy = np.sort(np.random.choice(a=range(p), 
                                    size=np.int32(p*col_sample), 
                                    replace=False), 
                   kind='quicksort')
      res['col_index_i'][iter] = iy                     
      X_iy = np.asarray(dtype=np.float64, a=X_)[:, iy] # must be X_!
      if res['weights_distr' ] == "uniform":
        W_i = np.random.rand(X_iy.shape[1], n_hidden_features)
      else: 
        W_i = np.random.randn(X_iy.shape[1], n_hidden_features)
      hhidden_layer_i = dropout_func(x=activation_choice(activation)(safe_sparse_dot(np.asarray(dtype=np.float64, a=X_iy), np.asarray(dtype=np.float64, a=W_i), backend)), 
                                     drop_prob=dropout, seed=seed)
      hh_i = np.hstack((X_iy, hhidden_layer_i)) if direct_link else hhidden_layer_i
      
      if row_sample < 1:
      
        ix = np.sort(np.random.choice(a=range(n), 
                                    size=np.int32(n*row_sample), 
                                    replace=False), 
                     kind='quicksort')
        X_iy_ix = X_iy[ix,:]       
        hidden_layer_i = dropout_func(x=activation_choice(activation)(safe_sparse_dot(X_iy_ix, W_i, backend)), 
                                      drop_prob=dropout, seed=seed)
        h_i =  np.hstack((X_iy_ix, hidden_layer_i)) if direct_link else hidden_layer_i        
        fit_obj.fit(X = np.asarray(dtype=np.float64, a=h_i), y = np.asarray(dtype=np.float64, a=e)[ix])

      else:
      
        fit_obj.fit(X = np.asarray(dtype=np.float64, a=hh_i), y = np.asarray(dtype=np.float64, a=e))
            
      e = e - learning_rate*np.asarray(dtype=np.float64, a=fit_obj.predict(np.asarray(dtype=np.float64, a=hh_i)))

      res['W_i'][iter] = np.asarray(dtype=np.float64, a=W_i)
      
      res['fit_obj_i'][iter] = deepcopy(fit_obj)

      current_error = np.linalg.norm(e)

      res['loss'].append(current_error)      

      try:              
        if np.abs(np.flip(np.diff(res['loss'])))[0] <= tolerance:
          res['n_estimators'] = iter
          break      
      except:
        pass
      
  return res
  
# 2 - 2 predict regressor -----

def predict_booster_regressor(object obj, double[:,::1] X, str backend):

  cdef int iter, n_estimators, n_classes
  cdef double learning_rate
  cdef double[:] preds_sum
  cdef double[:,::1] hh_i

  direct_link = obj['direct_link']
  n_estimators = obj['n_estimators']
  learning_rate = obj['learning_rate']
  activation = obj['activation']
  X_ = (X - obj['xm'][None, :])/obj['xsd'][None, :]
  preds_sum = np.zeros(X.shape[0])
  
  for iter in range(n_estimators):
  
    iy = obj['col_index_i'][iter]
    X_iy = X_[:, iy] # must be X_!
    W_i = obj['W_i'][iter]
    hh_i = np.hstack((X_iy, activation_choice(activation)(safe_sparse_dot(np.asarray(dtype=np.float64, a=X_iy), np.asarray(dtype=np.float64, a=W_i), backend)))) if direct_link else activation_choice(activation)(safe_sparse_dot(np.asarray(dtype=np.float64, a=X_iy), np.asarray(dtype=np.float64, a=W_i), backend))        
    preds_sum = preds_sum + learning_rate*np.asarray(dtype=np.float64, a=obj['fit_obj_i'][iter].predict(np.asarray(dtype=np.float64, a=hh_i)))
  
  return np.asarray(dtype=np.float64, a=obj['ym'] + np.asarray(dtype=np.float64, a=preds_sum))

# 2 - 3 update -----

def update_booster(object obj, double[:] X, y, double alpha=0.5, backend="cpu"):

  cdef int iter, n_estimators, n_classes, n_obs
  cdef double learning_rate
  cdef double[:] xm_old
  cdef double[:,::1] hh_i
  cdef str type_fit   

  n_obs = obj['n_obs']
  direct_link = obj['direct_link']
  n_estimators = obj['n_estimators']
  learning_rate = obj['learning_rate']
  activation = obj['activation']
  X_ = (X - obj['xm'][None, :])/obj['xsd'][None, :]
  
  if np.issubdtype(y.dtype, np.int64): # classification
    n_classes = obj["n_classes"]
    preds_sum = np.zeros(n_classes)
    Y = one_hot_encode2(y, n_classes)
    centered_y = Y - obj['Ym']
    residuals_i = np.zeros(n_classes)
    type_fit = "classification"
  else: # regression
    preds_sum = 0
    centered_y = y - obj['ym']
    residuals_i = 0
    type_fit = "regression"
  
  if type_fit == "regression": 
    #for iter in range(n_estimators):    
    #  iy = obj['col_index_i'][iter]
    #  X_iy = np.asarray(dtype=np.float64, a=X_[:, iy]).reshape(1, -1) # must be X_!
    #  W_i = obj['W_i'][iter]
    #  hh_i = np.hstack((X_iy, activation_choice(activation)(safe_sparse_dot(np.asarray(dtype=np.float64, a=X_iy), np.asarray(dtype=np.float64, a=W_i), backend)))) if direct_link else activation_choice(activation)(safe_sparse_dot(np.asarray(dtype=np.float64, a=X_iy), np.asarray(dtype=np.float64, a=W_i), backend))            
    #  preds_sum = preds_sum + learning_rate*np.asarray(dtype=np.float64, a=obj['fit_obj_i'][iter].predict(np.asarray(dtype=np.float64, a=hh_i)))
    #  residuals_i = centered_y - preds_sum
    #  obj['fit_obj_i'][iter].coef_ = np.asarray(dtype=np.float64, a=obj['fit_obj_i'][iter].coef_).ravel() + (n_obs**(-alpha))*safe_sparse_dot(residuals_i, hh_i, backend).ravel()    
    # Initialize cumulative sum of coefficients and count of iterations
    cumulative_coef_ = None

    for iter in range(n_estimators):    
        iy = obj['col_index_i'][iter]
        X_iy = np.asarray(dtype=np.float64, a=X_[:, iy]).reshape(1, -1)  # must be X_!
        W_i = obj['W_i'][iter]
        hh_i = (
            np.hstack((X_iy, activation_choice(activation)(safe_sparse_dot(np.asarray(dtype=np.float64, a=X_iy), np.asarray(dtype=np.float64, a=W_i), backend))))
            if direct_link
            else activation_choice(activation)(safe_sparse_dot(np.asarray(dtype=np.float64, a=X_iy), np.asarray(dtype=np.float64, a=W_i), backend))
        )
        
        preds_sum = preds_sum + learning_rate * np.asarray(dtype=np.float64, a=obj['fit_obj_i'][iter].predict(np.asarray(dtype=np.float64, a=hh_i)))
        residuals_i = centered_y - preds_sum
        
        # Update the coefficients as in your original code
        obj['fit_obj_i'][iter].coef_ = np.asarray(dtype=np.float64, a=obj['fit_obj_i'][iter].coef_).ravel() + (n_obs ** -alpha) * safe_sparse_dot(residuals_i, hh_i, backend).ravel()
        
        # If this is the first iteration, initialize cumulative_coef_ to the current coef_
        if cumulative_coef_ is None:
            cumulative_coef_ = obj['fit_obj_i'][iter].coef_.copy()
        else:
            cumulative_coef_ += obj['fit_obj_i'][iter].coef_

        # Calculate the running average of the coefficients and update obj['fit_obj_i'][iter].coef_
        obj['fit_obj_i'][iter].coef_ = cumulative_coef_ / (iter + 1)

  else: # type_fit == "classification": 

    # Initialize a variable to keep track of the cumulative sum of coef_
    cumulative_coef_sum = np.zeros_like(obj['fit_obj_i'][0].coef_)  # assuming all coef_ have the same shape

    for iter in range(n_estimators):    
        iy = obj['col_index_i'][iter]
        X_iy = np.asarray(dtype=np.float64, a=X_)[:, iy]  # must be X_!  
        W_i = obj['W_i'][iter]      
        gXW = np.asarray(dtype=np.float64, a=activation_choice(activation)(safe_sparse_dot(np.asarray(dtype=np.float64, a=X_iy), np.asarray(dtype=np.float64, a=W_i), backend)))
        
        if direct_link:
            hh_i = np.hstack((np.array(X_iy), np.array(gXW)))  
        else: 
            hh_i = gXW      

        preds_sum = preds_sum + learning_rate * np.asarray(dtype=np.float64, a=obj['fit_obj_i'][iter].predict(np.asarray(dtype=np.float64, a=hh_i)))            
        residuals_i = centered_y - preds_sum      
        
        # Update cumulative sum of coef_
        cumulative_coef_sum += np.asarray(dtype=np.float64, a=obj['fit_obj_i'][iter].coef_) 
        
        # Calculate the average of coef_ values so far
        average_coef = cumulative_coef_sum / (iter + 1)
        
        # Update coef_ with the average of all previous coef_ values + (n_obs ** (-alpha)) * safe_sparse_dot(residuals_i.T, hh_i, backend)
        obj['fit_obj_i'][iter].coef_ = average_coef 

  xm_old = obj['xm']
  obj['xm'] = (n_obs*np.asarray(dtype=np.float64, a=xm_old) + X)/(n_obs + 1)
  obj['xsd'] = np.sqrt(((n_obs - 1)*(obj['xsd']**2) + (np.asarray(dtype=np.float64, a=X) -np.asarray(dtype=np.float64, a=xm_old))*(np.asarray(dtype=np.float64, a=X) - obj['xm']))/n_obs)  
  obj['n_obs'] = n_obs + 1
  if type_fit == "regression":      
    obj['ym'] = (n_obs*obj['ym'] + y)/(n_obs + 1)    
  else: # type_fit == "classification"
    obj['Ym'] = (n_obs*obj['Ym'] + Y)/(n_obs + 1)
  
  return obj

# adapted from sklearn.utils.exmath
def safe_sparse_dot(a, b, backend="cpu", dense_output=False):
    """Dot product that handle the sparse matrix case correctly

    Parameters
    ----------
    a : array or sparse matrix
    b : array or sparse matrix
    dense_output : boolean, (default=False)
        When False, ``a`` and ``b`` both being sparse will yield sparse output.
        When True, output will always be a dense array.

    Returns
    -------
    dot_product : array or sparse matrix
        sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
    """    
    if backend in ("gpu", "tpu"):
        # modif when jax.scipy.sparse available
        return safe_sparse_dot(device_put(a), device_put(b)).block_until_ready()

    #    if backend == "cpu":
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = safe_sparse_dot(a, b)
    else:
        try:
            ret = a @ b
        except:
            ret = safe_sparse_dot(a, b)

    if (
        sparse.issparse(a)
        and sparse.issparse(b)
        and dense_output
        and hasattr(ret, "toarray")
    ):
        return ret.toarray()

    return ret

