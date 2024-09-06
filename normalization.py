import numpy as np

def normalization(X, lower, upper, MAX=None, MIN=None):
    n = X.shape[1]
    if MAX is None and MIN is None:
        Y = np.zeros_like(X)
        for i in range(n):
            Y[:, i] = (X[:, i] - np.min(X[:, i])) / (np.max(X[:, i]) - np.min(X[:, i])) * (upper - lower) + lower
        MIN_scale = np.min(X, axis=0)
        MAX_scale = np.max(X, axis=0)
    else:
        Y = np.zeros_like(X)
        for i in range(n):
            Y[:, i] = (X[:, i] - MIN[i]) / (MAX[i] - MIN[i]) * (upper - lower) + lower
    return Y, MAX_scale, MIN_scale