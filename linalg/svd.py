import numpy as np
from .bidiagonal import bidiagonalize
from .bidiag_svd import bidiag_svd
from .svd_utils import fix_signs


def svd(A, want="V", k=None):
    """
    Public wrapper for the SVD algorithm.
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    transposed = False

    # Work on tall matrixes
    if m < n:
        A = A.T
        m, n = n, m
        transposed = True

    need_u = False
    need_v = False
    if want == "both":
        need_u = need_v = True
    elif want == "U":
        need_u = not transposed
        need_v = transposed
    elif want == "V":
        need_u = transposed
        need_v = not transposed

    Ub, d, f, Vb = bidiagonalize(A, compute_u=need_u, compute_v=need_v)

    # Left/Right accumulator initialization
    U2 = np.eye(n) if need_u else None
    V2 = np.eye(n) if need_v else None

    bidiag_svd(d, f, U2, V2)

    U = Ub[:, :n] @ U2 if need_u else None
    V = Vb @ V2 if need_v else None

    # Single values
    s = d.copy()
    fix_signs(s, U, V)

    # Descending order of single values
    order = np.argsort(s)[::-1]
    s = s[order]

    if U is not None:
        U = U[:, order]
    if V is not None:
        V = V[:, order]

    if k is not None:
        s = s[:k]
        if U is not None:
            U = U[:, :k]
        if V is not None:
            V = V[:, :k]

    if transposed:
        U, V = V, U

    if want == "both":
        return U, s, V
    elif want == "U":
        return U, s
    elif want == "V":
        return V, s
    else:
        return s
