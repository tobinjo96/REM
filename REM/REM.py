"""Gaussian Mixture Model."""

#Modified code from Scikit-Learn 

import numpy as np
from scipy import linalg
from sklearn.metrics.pairwise import euclidean_distances
import scipy.spatial.distance as distance
from sklearn.neighbors import KernelDensity, KDTree
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings

import importlib.util
import sys
from GaussianMixture import GaussianMixture


import rpy2.robjects as robjects
from rpy2.robjects.packages import STAP

with open('REM/Overlap.R', 'r') as f:
    string = f.read()

overlap = STAP(string, "overlap")

###############################################################################
# Input parameter checkers used by the REM class
def density_broad_search_star(a_b):
  try:
    return euclidean_distances(a_b[1],a_b[0])
  except Exception as e:
    raise Exception(e)

def _estimate_density_distances(X, bandwidth):
  
  n_samples, n_features = X.shape
  
  if bandwidth == "spherical":
    
    center = X.sum(0)/n_samples
    
    X_centered = X - center
    
    covariance_data = np.einsum('ij,ki->jk', X_centered, X_centered.T) / (n_samples - 1)
    
    bandwidth = 1/(100 * n_features)*np.trace(covariance_data)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
    
    density = kde.score_samples(X)
  
  elif bandwidth == "diagonal":
    bandwidths = np.array([0.01*np.std(X[:, i]) for i in range(n_features)])
    
    var_type = 'c' * n_features
    
    dens_u = sm.nonparametric.KDEMultivariate(data = X, var_type = var_type, bw = bandwidths)
    
    density = dens_u.pdf(X)
  
  elif bandwidth == "normal_reference":
    var_type = 'c' * n_features
    
    dens_u = sm.nonparametric.KDEMultivariate(data = X, var_type = var_type, bw = 'normal_reference')
    
    density = dens_u.pdf(X)
  
  elif isinstance(bandwidth, int):
    kdt = KDTree(X, metric='euclidean')
    
    distances, neighbors = kdt.query(X, int(bandwidth))
    
    density = 1 / distances[:, int(bandwidth) - 1]
  
  elif isinstance(bandwidth, float):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
    
    density = kde.score_samples(X)
  
  
  kdt = KDTree(X, metric='euclidean')
  
  distances, neighbors = kdt.query(X, np.floor(np.sqrt(n_samples)).astype(int))
  
  best_distance = np.empty((X.shape[0]))
  
  radius_diff = density[:, np.newaxis] - density[neighbors]
  
  rows, cols = np.where(radius_diff < 0)
  
  rows, unidx = np.unique(rows, return_index =  True)
  
  cols = cols[unidx]
  
  best_distance[rows] = distances[rows, cols]
  
  search_idx = list(np.setdiff1d(list(range(X.shape[0])), rows))
  
  search_density = density[search_idx]
  
  GT_radius =  density > search_density[:, np.newaxis] 
  
  if any(np.sum(GT_radius, axis = 1) == 0):
    max_i = [i for i in range(GT_radius.shape[0]) if np.sum(GT_radius[i,:]) == 0]
    
    if len(max_i) > 1:
      for max_j in max_i[1:len(max_i)]:
        GT_radius[max_j, search_idx[max_i[0]]] = True
    
    max_i = max_i[0]
    
    best_distance[search_idx[max_i]] = np.sqrt(((X - X[search_idx[max_i], :])**2).sum(1)).max()
    
    GT_radius = np.delete(GT_radius, max_i, 0)
  
    del search_idx[max_i]
  
  GT_distances = ([X[search_idx[i], np.newaxis], X[GT_radius[i, :], :]] for i in range(len(search_idx)))
  
  distances_bb = list(map(density_broad_search_star, list(GT_distances)))
  
  argmin_distance = [np.argmin(l) for l in distances_bb]
  
  for i in range(GT_radius.shape[0]):
    best_distance[search_idx[i]] = distances_bb[i][argmin_distance[i]]
  
  return density, best_distance

def _create_decision_plots(density, distance):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(density, distance, s = 5, alpha = 1, color='black')
    plt.xlabel("Log of Density")
    plt.ylabel("Distance to Neighbor of Higher Density")
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(density)), np.sort(density*distance), s = 5, alpha = 1, color='black')
    plt.xlabel("Index")
    plt.ylabel("Product of Density and Distance")
    plt.tight_layout()
    plt.savefig("Exemplar Selection Plots.pdf")
    plt.close()

def _select_exemplars(X, density, distance):
    print("Exemplars selection plots saved as 'Exemplar Selection Plots.pdf'.")
    
    method = input("Select exemplars from Plot 1 or Plot 2? (1/2)")
    
    if method not in ["1", "2"]:
      raise ValueError(
              "Invalid value entered: %s "
              "Value entered should be 1 or 2."
              %method
            )
    
    if method == "1":
      
      density_threshold = float(input("Threshold for log of local density: "))
      
      distance_threshold = float(input("Threshold for distance to neighbor of higher density: "))
      
      density_inlier = density > density_threshold
      
      distance_inlier = distance > distance_threshold
      
      means_idx = np.where(density_inlier * distance_inlier)[0]
      
      remainder_idx = np.where(~(density_inlier * distance_inlier))[0]
      
      means = X[means_idx, :]
      
      X_iter = X[remainder_idx, :]
    
    elif method == "2":
      
      product_threshold = float(input("Threshold for prodcut of density and distance: "))
      
      product_inlier = (density * distance) > product_threshold
      
      means_idx = np.where(product_inlier)[0]
      
      remainder_idx = np.where(~product_inlier)[0]
      
      means = X[means_idx, :]
      
      X_iter = X[remainder_idx, :]
    
    print ("%s means selected." %means.shape[0])
    return X_iter, means

def _select_exemplars_fromK(X, density, distance, max_components):
    
    n_samples, _ = X.shape
    
    means_idx = np.argsort( - density * distance)[range(max_components)]
    
    remainder_idx = np.argsort( - density * distance)[range(max_components, n_samples)]
    
    means = X[means_idx, :]
    
    X_iter = X[remainder_idx, :]
    
    print ("%s means selected." %means.shape[0])
    return X_iter, means


def _initialize_covariances(X, means, covariance_type):
    
    n_samples, n_features = X.shape
    
    n_components = means.shape[0]
    
    center = X.sum(0)/n_samples
    
    X_centered = X - center
    
    covariance_data = np.einsum('ij,ki->jk', X_centered, X_centered.T) / (n_samples - 1)
    
    variance = 1/(n_components * n_features)*np.trace(covariance_data) 
    
    if covariance_type == "full":
      covariances = np.stack([np.diag(np.ones(n_features) * variance) for _ in range(n_components)])
    elif covariance_type == "spherical":
      covariances = np.repeat(variance, n_components)
    elif covariance_type == "tied":
      covariances = np.diag(np.ones(n_features) * variance)
    elif covariance_type == "diag":
      covariances = np.ones((n_components, n_features)) * variance
    
    return covariances
   
def _check_weights(weights, n_components):
    """Check the user provided 'weights'.
    
    Parameters
    ----------
    weights : array-like of shape (n_components,)
        The proportions of components of each mixture.
    
    n_components : int
        Number of components.
    
    Returns
    -------
    weights : array, shape (n_components,)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(weights, (n_components,), "weights")
    
    # check range
    if any(np.less(weights, 0.0)) or any(np.greater(weights, 1.0)):
        raise ValueError(
            "The parameter 'weights' should be in the range "
            "[0, 1], but got max value %.5f, min value %.5f"
            % (np.min(weights), np.max(weights))
        )
    
    # check normalization
    if not np.allclose(np.abs(1.0 - np.sum(weights)), 0.0):
        raise ValueError(
            "The parameter 'weights' should be normalized, but got sum(weights) = %.5f"
            % np.sum(weights)
        )
    return weights


def _check_means(means, n_components, n_features):
    """Validate the provided 'means'.
    
    Parameters
    ----------
    means : array-like of shape (n_components, n_features)
        The centers of the current components.
    
    n_components : int
        Number of components.
    
    n_features : int
        Number of features.
    
    Returns
    -------
    means : array, (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_components, n_features), "means")
    return means


def _check_precision_positivity(precision, covariance_type):
    """Check a precision vector is positive-definite."""
    if np.any(np.less_equal(precision, 0.0)):
        raise ValueError("'%s precision' should be positive" % covariance_type)


def _check_precision_matrix(precision, covariance_type):
    """Check a precision matrix is symmetric and positive-definite."""
    if not (
        np.allclose(precision, precision.T) and np.all(linalg.eigvalsh(precision) > 0.0)
    ):
        raise ValueError(
            "'%s precision' should be symmetric, positive-definite" % covariance_type
        )


def _check_precisions_full(precisions, covariance_type):
    """Check the precision matrices are symmetric and positive-definite."""
    for prec in precisions:
        _check_precision_matrix(prec, covariance_type)


def _check_precisions(precisions, covariance_type, n_components, n_features):
    """Validate user provided precisions.
    
    Parameters
    ----------
    precisions : array-like
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)
    
    covariance_type : str
    
    n_components : int
        Number of components.
    
    n_features : int
        Number of features.
    
    Returns
    -------
    precisions : array
    """
    precisions = check_array(
        precisions,
        dtype=[np.float64, np.float32],
        ensure_2d=False,
        allow_nd=covariance_type == "full",
    )
    
    precisions_shape = {
        "full": (n_components, n_features, n_features),
        "tied": (n_features, n_features),
        "diag": (n_components, n_features),
        "spherical": (n_components,),
    }
    _check_shape(
        precisions, precisions_shape[covariance_type], "%s precision" % covariance_type
    )
    
    _check_precisions = {
        "full": _check_precisions_full,
        "tied": _check_precision_matrix,
        "diag": _check_precision_positivity,
        "spherical": _check_precision_positivity,
    }
    _check_precisions[covariance_type](precisions, covariance_type)
    return precisions


###############################################################################
# Gaussian mixture parameters estimators (used by the M-Step)


def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    """Estimate the full covariance matrices.
    
    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)
    
    X : array-like of shape (n_samples, n_features)
    
    nk : array-like of shape (n_components,)
    
    means : array-like of shape (n_components, n_features)
    
    reg_covar : float
    
    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[:: n_features + 1] += reg_covar
    return covariances


def _estimate_gaussian_covariances_tied(resp, X, nk, means, reg_covar):
    """Estimate the tied covariance matrix.
    
    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)
    
    X : array-like of shape (n_samples, n_features)
    
    nk : array-like of shape (n_components,)
    
    means : array-like of shape (n_components, n_features)
    
    reg_covar : float
    
    Returns
    -------
    covariance : array, shape (n_features, n_features)
        The tied covariance matrix of the components.
    """
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(nk * means.T, means)
    covariance = avg_X2 - avg_means2
    covariance /= nk.sum()
    covariance.flat[:: len(covariance) + 1] += reg_covar
    return covariance


def _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
    """Estimate the diagonal covariance vectors.
    
    Parameters
    ----------
    responsibilities : array-like of shape (n_samples, n_components)
    
    X : array-like of shape (n_samples, n_features)
    
    nk : array-like of shape (n_components,)
    
    means : array-like of shape (n_components, n_features)
    
    reg_covar : float
    
    Returns
    -------
    covariances : array, shape (n_components, n_features)
        The covariance vector of the current components.
    """
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means ** 2
    avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar


def _estimate_gaussian_covariances_spherical(resp, X, nk, means, reg_covar):
    """Estimate the spherical variance values.
    
    Parameters
    ----------
    responsibilities : array-like of shape (n_samples, n_components)
    
    X : array-like of shape (n_samples, n_features)
    
    nk : array-like of shape (n_components,)
    
    means : array-like of shape (n_components, n_features)
    
    reg_covar : float
    
    Returns
    -------
    variances : array, shape (n_components,)
        The variance values of each components.
    """
    return _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar).mean(1)


def _estimate_gaussian_parameters(X, resp, reg_covar, covariance_type):
    """Estimate the Gaussian distribution parameters.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data array.
    
    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X.
    
    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.
    
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.
    
    Returns
    -------
    nk : array-like of shape (n_components,)
        The numbers of data samples in the current components.
    
    means : array-like of shape (n_components, n_features)
        The centers of the current components.
    
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    #means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = {
        "full": _estimate_gaussian_covariances_full,
        "tied": _estimate_gaussian_covariances_tied,
        "diag": _estimate_gaussian_covariances_diag,
        "spherical": _estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar)
    return nk, covariances


def _compute_precision_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the precisions.
    
    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.
    
    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar."
    )
    
    if covariance_type == "full":
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(
                cov_chol, np.eye(n_features), lower=True
            ).T
    elif covariance_type == "tied":
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(
            cov_chol, np.eye(n_features), lower=True
        ).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1.0 / np.sqrt(covariances)
    return precisions_chol

class REM:
    def __init__(
        self,
        *,
        covariance_type="full",
        criteria = "none",
        max_components = None,
        bandwidth = "diagonal",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        random_state=None,
        verbose=0,
        verbose_interval=10,
    ):
        self.covariance_type = covariance_type
        self.criteria = criteria
        self.max_components = max_components
        self.bandwidth = bandwidth
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.verbose_interval = verbose_interval
    
    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        _, n_features = X.shape
        if self.covariance_type not in ["spherical", "tied", "diag", "full"]:
            raise ValueError(
                "Invalid value for 'covariance_type': %s "
                "'covariance_type' should be in "
                "['spherical', 'tied', 'diag', 'full']"
                % self.covariance_type
            )
        
        if self.criteria not in ["none", "aic", "bic", "icl"]:
            raise ValueError(
                "Invalid value for 'criteria': %s "
                "'criteria' should be in "
                "['none', 'aic', 'bic', 'icl']"
                % self.criteria
            )
        
        if self.bandwidth != "diagonal" or "spherical" or "normal_reference" or isinstance(bandwidth, int) or isinstance(bandwidth, float):
            raise ValueError(
              "Invalid value for 'bandwidth': %s"
              "'bandwidth' should be 'diagonal', 'spherical', 'normal_reference' or a float."
              %self.bandwidth
            )
    
    def _initialize_parameters(self, X):
        """Initialization of the Gaussian mixture exemplars from a decision plot.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        
        """
        n_samples, _ = X.shape
        
        density, distances = _estimate_density_distances(X, self.bandwidth)
        
        if self.max_components is None:
          _create_decision_plots(density, distances)
          
          self.X_iter, self.means_iter = _select_exemplars(X, density, distances)
        else:
          self.X_iter, self.means_iter = _select_exemplars_fromK(X, density, distances, self.max_components)
        
        self.n_components_iter = self.means_iter.shape[0]
        
        self.covariances_iter = _initialize_covariances(X, self.means_iter, self.covariance_type)
        
        self.weights_iter = np.ones((self.n_components_iter))/self.n_components_iter
        
        self.mixtures = [GaussianMixture.GaussianMixture(n_components = self.n_components_iter, weights = self.weights_iter, means = self.means_iter, covariances = self.covariances_iter, covariance_type = self.covariance_type).fit(self.X_iter)]
        
        self.n_mixtures = 1
    
    def compute_overlap(self, n_features):
        # 
        # n_components = self.n_components_iter
        # weights = self.weights_iter
        # means = self.means_iter
        # covariances = self.covariances_iter
        # 
        # eps = 1e-6
        # lim = int(1e6)
        # 
        # n_features_c = ctypes.c_int(n_features)
        # n_components_c = ctypes.c_int(n_components)
        # weights = weights.astype(np.float64)
        # means1 = means.flatten()
        # means1 = means1.astype(np.float64)
        # covariances1 = covariances.flatten()
        # covariances1 = covariances1.astype(np.float64)
        # pars = np.array([eps, eps]).astype(np.float64)
        # lim = ctypes.c_int(lim)
        # OmegaMap1 = np.zeros(n_components ** 2).astype(np.float64)
        # BarOmega = ctypes.c_double(1)
        # MaxOmega = ctypes.c_double(1)
        # rcMax = np.array([0, 0]).astype(int)
        # 
        # 
        # n_features_ptr = ctypes.byref(n_features_c)
        # n_components_ptr = ctypes.byref(n_components_c)
        # weights_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # means1_ptr = means1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # covariances1_ptr = covariances1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # pars_ptr = pars.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # lim_ptr = ctypes.byref(lim)
        # OmegaMap1_ptr = OmegaMap1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # BarOmega_ptr = ctypes.byref(BarOmega)
        # MaxOmega_ptr = ctypes.byref(MaxOmega)
        # rcMax_ptr = rcMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        # 
        # overlap.runExactOverlap.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int))
        # overlap.runExactOverlap.restypes = None
        # 
        # overlap.runExactOverlap(n_features_ptr, n_components_ptr, weights_ptr, means1_ptr, covariances1_ptr, pars_ptr, lim_ptr, OmegaMap1_ptr, BarOmega_ptr, MaxOmega_ptr, rcMax_ptr)
        # 
        
        covariances_jitter = np.zeros(self.covariances_iter.shape)
        for i in range(self.n_components_iter):
          val, vec = np.linalg.eig(self.covariances_iter[i])
          val += np.abs(np.random.normal(loc = 0, scale = 0.01, size = n_features))
          covariances_jitter[i, :, :] = vec.dot(np.diag(val)).dot(np.linalg.inv(vec))
                
        while True:
          weights1 = robjects.FloatVector([i for i in self.weights_iter])
          means1 = robjects.FloatVector([i for i in self.means_iter.flatten()])
          covariances1 = robjects.FloatVector([i for i in covariances_jitter.flatten()])
          OmegaMap1 = overlap.overlap(Pi = weights1, Mu = means1, S = covariances1)
          
          OmegaMap1 = np.reshape(OmegaMap1, (self.n_components_iter, self.n_components_iter))
          
          OmegaMap1 -= np.diag(np.ones(self.n_components_iter))
          
          if np.max(OmegaMap1.max(1)) > 0:
            break
          else:
            covariances_jitter *= 1.1
        
        return OmegaMap1.max(1)
    
    def compute_theta(self, distances, covariances_logdet_penalty, overlap_max):
        
        n_samples, _ = self.X_iter.shape
        
        thetas = np.ones(self.n_components_iter) * np.nan
        
        for i in range(self.n_components_iter):
          P = distances + covariances_logdet_penalty - (distances[:, i] + covariances_logdet_penalty[i])[:, np.newaxis]
          
          Ranges = P/(overlap_max[i] - overlap_max)
          
          noni_idx = list(range(self.n_components_iter))
          noni_idx.pop(i)
          
          overlap_noni = overlap_max[noni_idx]
          
          Ranges = Ranges[:, noni_idx]
          
          ltidx = np.where(overlap_max[i] < overlap_noni)[0]
          
          gtidx = np.where(overlap_max[i] > overlap_noni)[0]
          
          raw_intervals = []
          union_intervals = []
          for s in range(n_samples):
            raw_intervals.append([])
            union_intervals.append([])
            for t in ltidx:
              raw_intervals[s].append((-np.inf, Ranges[s, t]))
            
            for t in gtidx:
              raw_intervals[s].append((Ranges[s, t], np.inf))
            
            for begin, end in sorted(raw_intervals[s]):
              if union_intervals[s] and union_intervals[s][-1][1] >= begin - 1:
                  union_intervals[s][-1][1] = max(union_intervals[s][-1][1], end)
              
              else:
                  union_intervals[s].append([begin, end])
          
          union_intervals = [item for sublist in union_intervals for item in sublist]
          
          start, end = union_intervals.pop()
          while union_intervals:
             start_temp, end_temp = union_intervals.pop()
             start = max(start, start_temp)
             end = min(end, end_temp)
          
          if start < end and start > 0:
            thetas[i] = start
        
        theta = thetas[~np.isnan(thetas)].min()
        
        return theta * 1.0001
        
    def return_refined(self, resps):
        
        self.weights_iter = resps.sum(0)/resps.shape[0]
        
        self.weights_iter[self.weights_iter < 0.00001] = 0
        
        rm_means = self.means_iter[self.weights_iter == 0,:]
        
        if rm_means.ndim == 1:
          self.X_iter = np.append(self.X_iter, rm_means[:, np.newaxis], axis = 0)
        else:
          self.X_iter = np.append(self.X_iter, rm_means, axis = 0)
        
        self.means_iter = self.means_iter[self.weights_iter != 0,:]
        
        self.covariances_iter = self.covariances_iter[self.weights_iter != 0, :, :]
        
        self.weights_iter = self.weights_iter[self.weights_iter != 0]
        
        self.n_components_iter = len(self.weights_iter)
    
    def prune_exemplar(self):
        
        n_samples, n_features = self.X_iter.shape
        
        distances = np.zeros((n_samples, self.n_components_iter))
        
        resp = np.ones((n_samples, self.n_components_iter))/self.n_components_iter
        
        self.covariances_iter += np.ones(self.covariances_iter.shape)*1e-6
        
        for j in range(self.n_components_iter):
          distances[:,j,np.newaxis] = distance.cdist(self.X_iter, self.means_iter[j,:][np.newaxis], metric='mahalanobis', VI=np.linalg.inv(self.covariances_iter[j, :, :]))
        
        covariances_logdet_penalty = np.array([np.log(np.linalg.det(self.covariances_iter[i])) for i in range(self.n_components_iter) ])/n_samples
        
        overlap_max = self.compute_overlap(n_features)
        
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          
          theta = self.compute_theta(distances, covariances_logdet_penalty, overlap_max)
        
        Ob = distances + covariances_logdet_penalty + (theta * overlap_max)
        
        s_min = Ob.argmin(1)
        
        resps = np.zeros((n_samples, self.n_components_iter))
        
        resps[range(resps.shape[0]), s_min] = 1
        
        self.return_refined(resps)
    
    def fit(self, X, y = None):
        
        self.fit_predict(X, y)
        
        return self
    
    def fit_predict(self, X, y = None):
        #X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        
        self._initialize_parameters(X)
        
        while self.n_components_iter > 1:
          
          self.prune_exemplar()
          
          mixture = GaussianMixture.GaussianMixture(n_components = self.n_components_iter, weights = self.weights_iter, means = self.means_iter, covariances = self.covariances_iter, covariance_type = self.covariance_type).fit(self.X_iter)
          
          self.weights_iter = mixture.weights_
          
          self.covariances_iter = mixture.covariances_
          
          self.mixtures.append(mixture)
        
        if self.criteria == "aic":
            self.aics_ = []
            
            for mixture in self.mixtures:
              self.aics_.append(mixture.aic(X))
            
            self.aic_mixture = self.mixtures[np.argmax(self.aics_)]
        
        elif self.criteria == "bic":
            self.bics_ = []
            for mixture in self.mixtures:
              self.bics_.append(mixture.bic(X))
            
            self.bic_mixture = self.mixtures[np.argmax(self.bics_)]
        
        elif self.criteria == "icl":
            self.icls_ = []
            for mixture in self.mixtures:
              self.icls_.append(mixture.icl(X))
            
            self.icl_mixture = self.mixtures[np.argmax(self.icls_)]
        
        elif self.criteria == "all":
            self.aics_ = []
            
            for mixture in self.mixtures:
              self.aics_.append(mixture.aic(X))
            
            self.aic_mixture = self.mixtures[np.argmin(self.aics_)]
            
            self.bics_ = []
            for mixture in self.mixtures:
              self.bics_.append(mixture.bic(X))
            
            self.bic_mixture = self.mixtures[np.argmin(self.bics_)]
            
            self.icls_ = []
            for mixture in self.mixtures:
              self.icls_.append(mixture.icl(X))
            
            self.icl_mixture = self.mixtures[np.argmin(self.icls_)]
        
            
