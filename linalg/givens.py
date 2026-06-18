import numpy as np


def givens(a, b):
    a = np.float64(a)
    b = np.float64(b)

    if b == 0.0:
        return np.float64(1.0), np.float64(0.0)

    if np.abs(b) > np.abs(a):
        tau = -a / b
        s = 1.0 / np.sqrt(1.0 + tau**2)
        c = s * tau
    else:
        tau = -b / a
        c = 1.0 / np.sqrt(1.0 + tau**2)
        s = c * tau

    return c, s
