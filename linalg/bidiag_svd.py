import numpy as np
from .svd_utils import deflate, find_block
from .svd_core import zero_out_block_fallback, gk_step


def bidiag_svd(d, f, U2, V2):
    """
    Orchestrates the iterative diagonalization of the bidiagonal matrix.
    Reference: Algorithm 8.6.2 Golub & Van Loan
    """
    n = len(d)
    EPS = np.finfo(np.float64).eps
    max_iter = 100 * n
    iters = 0

    while True:
        deflate(d, f)

        bounds = find_block(f, n)
        if bounds is None:
            break
        lo, hi = bounds

        max_diag = np.max(np.abs(d[lo : hi + 1]))
        zero_at = None
        for i in range(lo, hi + 1):
            if abs(d[i]) <= EPS * max_diag:
                zero_at = i
                break

        if zero_at is not None:
            zero_out_block_fallback(d, f, U2, V2, lo, hi)
            continue

        gk_step(d, f, U2, V2, lo, hi)

        iters += 1
        if iters > max_iter:
            raise RuntimeError(f"SVD failed to converge after {max_iter} iterations.")
