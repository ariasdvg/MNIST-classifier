import numpy as np


def house(x):
    """Given vector x, return the Householder reflection"""
    x = np.asarray(x, dtype=np.float64)
    sigma = np.dot(x[1:], x[1:])
    v = np.empty_like(x)
    v[0] = 1.0
    v[1:] = x[1:]

    if sigma == 0.0:
        if x[0] >= 0.0:
            beta = 0.0
        else:
            beta = 2.0
    else:
        mu = np.sqrt(x[0] ** 2 + sigma)
        if x[0] <= 0.0:
            v0 = x[0] - mu
        else:
            v0 = -sigma / (x[0] + mu)
        beta = (2 * v0**2) / (sigma + v0**2)
        v[0] = v0
        v /= v[0]
    return v, beta
