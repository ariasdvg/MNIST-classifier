import numpy as np


def deflate(d, f):
    """
    DEFLATION: Zeroes out superdiagonal elements that are negligibly small.
    Operates in-place.
    """
    n = len(d)
    EPS = np.finfo(np.float64).eps
    for i in range(n - 1):
        if abs(f[i]) <= EPS * (abs(d[i]) + abs(d[i + 1])):
            f[i] = 0.0


def find_block(f, n):
    """
    FIND THE ACTIVE BLOCK: Scan to find the lowest unreduced block.
    Returns (lo, hi) inclusive indices of the block, or None.
    """

    # Find upper limit of the block
    hi = n - 1
    while hi > 0 and f[hi - 1] == 0.0:
        hi -= 1

    # SVD is complete
    if hi == 0:
        return None

    # Now find the lower limit
    lo = hi
    while lo > 0 and f[lo - 1] != 0.0:
        lo -= 1

    return (lo, hi)


def wilkinson_shift(d, f, lo, hi):
    """
    Computes the Wilkinson shift from the trailing 2x2 submatrix of B^T B.
    """
    d_m = d[hi - 1]
    f_m = f[hi - 1]
    d_n = d[hi]

    f_m_minus_1 = f[hi - 2] if hi - 2 >= lo else 0.0  # Check if out of bounds

    # 2x2 trailing matrix block
    y = d_m**2 + f_m_minus_1**2
    x = d_n**2 + f_m**2
    z = d_m * f_m

    w = (y - x) / 2.0
    sign = 1.0 if w >= 0.0 else -1.0

    denom = w + sign * np.sqrt(w**2 + z**2)
    if denom == 0.0:
        return x

    return x - (z**2) / denom


def fix_signs(s, U, V):
    """
    POST-PROCESS: Ensures all singular values are mathematically positive.
    Absorbs negative signs into singular vectors.
    """
    for i in range(len(s)):
        if s[i] < 0.0:
            s[i] = -s[i]
            if U is not None:
                U[:, i] = -U[:, i]
            elif V is not None:
                V[:, i] = -V[:, i]
