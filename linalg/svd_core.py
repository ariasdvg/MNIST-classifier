import warnings
import numpy as np
from .givens import givens


def zero_out_block_fallback(d, f, U2, V2, lo, hi):
    """
    Zero-diagonal special case fallback.
    Resolves singular blocks with numpy SVD and folds the result
    into the accumulators.
    """
    sz = hi - lo + 1
    B = np.zeros((sz, sz))
    for i in range(sz):
        B[i, i] = d[lo + i]
        if i < sz - 1:
            B[i, i + 1] = f[lo + i]

    warnings.warn(
        f"zero-diagonal in block [{lo}:{hi}]; using library SVD fallback for this block",
        RuntimeWarning,
    )

    Ub, sb, Vtb = np.linalg.svd(B)

    if U2 is not None:
        U2[:, lo : hi + 1] = U2[:, lo : hi + 1] @ Ub
    if V2 is not None:
        V2[:, lo : hi + 1] = V2[:, lo : hi + 1] @ Vtb.T

    for i in range(sz):
        d[lo + i] = sb[i]
        if i < sz - 1:
            f[lo + i] = 0.0


def gk_step(d, f, U2, V2, lo, hi):
    """
    ONE GOLUB-KAHAN STEP: Performs the implicit shifted QR step on B^T B.
    Chases the bulge down the diagonal from lo to hi using purely scalar algebra.

    Parameters
    ----------
    d: 1D-array with diagonal elements
    f: 1D-array with superdiagonal elements
    U2: left accumulator
    V2: right accumulator
    (lo,hi): indexes of active block

    Reference: Algorithm 8.6.1 Golub & Van Loan
    """
    from .svd_utils import wilkinson_shift

    mu = wilkinson_shift(d, f, lo, hi)

    y = d[lo] ** 2 - mu
    z = d[lo] * f[lo]

    for k in range(lo, hi):
        # Right rotation (acts on columns k, k+1)
        c, s = givens(y, z)

        if k > lo:
            f[k - 1] = c * y - s * z

        y0 = c * d[k] - s * f[k]
        f_k_prime = s * d[k] + c * f[k]

        z0 = -s * d[k + 1]  # subdiagonal bulge
        d_kp1_prime = c * d[k + 1]

        # Updating the right accumulator
        if V2 is not None:
            V_k = V2[:, k].copy()
            V_kp1 = V2[:, k + 1].copy()
            V2[:, k] = c * V_k - s * V_kp1
            V2[:, k + 1] = s * V_k + c * V_kp1

        # Left rotation (acts on rows k, k+1)
        c, s = givens(y0, z0)

        d[k] = c * y0 - s * z0
        y = c * f_k_prime - s * d_kp1_prime
        d[k + 1] = s * f_k_prime + c * d_kp1_prime

        if k < hi - 1:
            z = -s * f[k + 1]  # superdiagonal bulge
            f[k + 1] = c * f[k + 1]

        # Updating the left accumulator
        if U2 is not None:
            U_k = U2[:, k].copy()
            U_kp1 = U2[:, k + 1].copy()
            U2[:, k] = c * U_k - s * U_kp1
            U2[:, k + 1] = s * U_k + c * U_kp1

    f[hi - 1] = y
