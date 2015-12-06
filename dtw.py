"""Dynamic Time Warping (DTW) module.

`Please see this presentation from UIUC for an explanation. 
<http://luthuli.cs.uiuc.edu/~daf/courses/CS-498-DAF-PS/Lecture%2018%20-%20Time%20series,%20DTW.pdf>`_

Thanks to the dtw module for inspiration: https://github.com/pierre-rouanet/dtw
"""
import numpy as np


def l1(x, y):
    """Computes the L1 norm of two sequences."""
    return np.linalg.norm(x - y, ord=1)


def cost_matrix(x, y, dist_func=l1):
    """Computes the DTW cost matrix given two sequences.

    Parameters
    ----------
    x : ndarray
        N1 x M array
    y : ndarray
        N2 x M array
    dist_func : callable, optional
        Distance function to use in cost evaluation. Default: L1 norm.

    Returns
    -------
    cost : N1 x N2 array
        The accumulated cost matrix.
    """
    x = np.atleast_2d(x)
    n1 = len(x)
    y = np.atleast_2d(y)
    n2 = len(y)

    cost = np.empty((n1+1, n2+1), dtype=float)
    cost[0, 1:] = np.inf
    cost[1:, 0] = np.inf

    for i in range(n1):
        for j in range(n2):
            cost[i+1, j+1] = dist_func(x[i], y[j])

    for i in range(n1):
        for j in range(n2):
            cost[i+1, j+1] += min(cost[i, j], cost[i, j+1], cost[i+1, j])

    cost = cost[1:, 1:]
    return cost


def distance(x=None, y=None, cost=None, dist_func=l1):
    """Computes the DTW distance given either two sequences or a cost matrix.

    Parameters
    ----------
    x : ndarray, optional
        N1 x M array
    y : ndarray, optional
        N2 x M array
    cost : N1 x N2 array, optional
        The accumulated cost matrix.
    dist_func : callable, optional
        Distance function to use in cost evaluation. Default: L1 norm.

    Returns
    -------
    d : float
        The minimum distance between the series.
    """
    if cost is None:
        if x is None or y is None:
            raise ValueError("x & y cannont be None if cost is None!")
        cost = cost_matrix(x, y, dist_func=dist_func)
    return cost[-1, -1] / np.sum(cost.shape)


def warp_path(cost):
    """Computes the warp path from a cost matrix.

    Parameters
    ----------
    cost : N1 x N2 array
        The accumulated cost matrix.

    Returns
    -------
    w : M x 2 array
        The warp path.
    """
    i, j = cost.shape[0] - 1, cost.shape[1] - 1
    p, q = [i], [j]
    while (i > 0 and j > 0):
        prev = np.argmin((cost[i-1, j-1], cost[i-1, j], cost[i, j-1]))
        if (prev == 0):
            i -= 1
            j -= 1
        elif (prev == 1):
            i -= 1
        elif (prev == 2):
            j -= 1
        p.append(i)
        q.append(j)
    p.append(0)
    q.append(0)
    return np.array([p[::-1], q[::-1]])


def dtw(x, y, dist_func=l1):
    """Calculates the dynamic time warping of two sequences.

    Parameters
    ----------
    x : ndarray
        N1 x M array
    y : ndarray
        N2 x M array
    dist_func : callable, optional
        Distance function to use in cost evaluation. Default: L1 norm.

    Returns
    -------
    d : float
        The minimum distance between the series.
    cost : N1 x N2 array
        The accumulated cost matrix.
    w : M x 2 array
        The warp path.
    """
    cost = cost_matrix(x, y, dist_func=l1)
    d = distance(cost=cost)
    w = warp_path(cost)
    return d, cost, w


def distance_matrix(mfccs, callback=None):
    """Computes a distance matrix from a list mfccs"""
    n = len(mfccs)
    stat_numer = 0
    stat_denom = (n**2) / 2
    dists = np.empty((n, n), 'f8')
    for i in range(n):
        for j in range(i, n):
            # this matrix is symmetric by def.
            dists[i,j] = dists[j,i] = distance(mfccs[i], mfccs[j])
            if callback is not None:
                stat_numer += 1
                callback(stat_numer / stat_denom)
    return dists
