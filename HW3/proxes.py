# this file contains collections of proxes we learned in the class
import numpy as np
from scipy.optimize import bisect


# =============================================================================
# TODO Complete the following prox for simplex
# =============================================================================

# Prox of capped simplex
# -----------------------------------------------------------------------------


def prox_csimplex(z, k):
    """
    Prox of capped simplex
            argmin_x 1/2||x - z||^2 s.t. x in k-capped-simplex.

    input
    -----
    z : arraylike
            reference point
    k : int
            positive number between 0 and z.size, denote simplex cap

    output
    ------
    x : arraylike
            projection of z onto the k-capped simplex
    """
    # safe guard for k
    assert 0 <= k <= z.size, 'k: k must be between 0 and dimension of the input.'

    def derivative(v):
        grad = -k
        for i in range(z.size):
            grad += np.minimum(1, np.maximum(0, z[i] - v))
        return grad

    ## use scipy to find the v_star
    v_star = bisect(derivative, -10, 10)
    # X = np.zeros(z.size)
    # for i in range(z.size):
    #     X[i] = np.minimum(1, np.maximum(0, z[i] - v_star))
    X = np.minimum(1, np.maximum(0, z - v_star))
    return X


def prox_l1(sigma_Y, t):
    sigma_X = np.maximum(0, sigma_Y - t)
    return sigma_X


def rank_project(Y, k):
    U_y, sigma_y, Vt_y = np.linalg.svd(Y)
    sigma_X = sigma_y[0:k]
    sigma_X = np.hstack((sigma_X, np.zeros(len(sigma_y) - k)))  ## truncated singular values
    Sigma_X = np.diag(sigma_X)
    X = U_y @ Sigma_X @ Vt_y
    return X


def nuclear_prox(Y, t):
    """Nuclear norm proximal operator
    argmin_M 1/(2t)||M - Y||^2 + ||M||_{*}

    Parameters
    ----------
    Y : 2 dimensional array
    k : positive integer

    Returns
    -------
    2 dimensional array
            proximal operator applied to Y
    """
    U_y, sigma_y, Vt_y = np.linalg.svd(Y)
    sigma_X = prox_l1(sigma_y, t)
    Sigma_X = np.diag(sigma_X)
    X = U_y @ Sigma_X @ Vt_y

    return X
