import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_regression


def shannon_entropy(x, bins=10):
    """
    Computes the Shannon entropy (in bits) of a continuous variable by discretizing it.

    Parameters:
    -----------
    x : array-like
        Input data (1D array or Series).
    bins : int, optional (default=10)
        Number of bins to use for discretization.

    Returns:
    --------
    float
        Estimated Shannon entropy. Higher values indicate more disorder or information content.
    """
    x = np.asarray(x)
    est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    x_binned = est.fit_transform(x.reshape(-1, 1)).flatten()
    _, counts = np.unique(x_binned, return_counts=True)
    probs = counts / len(x_binned)
    return -np.sum(probs * np.log2(probs + 1e-12))  # 1e-12 avoids log(0)




def mutual_information(x, y, bins=10, seed=None):
    """
    Estimates the mutual information (in bits) between two continuous variables.

    Parameters:
    -----------
    x : array-like
        First variable (predictor).
    y : array-like
        Second variable (response).
    bins : int, optional (default=10)
        Number of bins for discretizing both variables.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns:
    --------
    float
        Estimated mutual information between x and y. Higher values indicate stronger dependency.
    """
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)
    
    est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    x_binned = est.fit_transform(x).flatten().reshape(-1, 1)
    y_binned = est.fit_transform(y.reshape(-1, 1)).flatten()
    
    return mutual_info_regression(x_binned, y_binned, random_state=seed)[0]





