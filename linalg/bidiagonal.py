import householder
import numpy as np


def bidiagonalize(A, compute_u=True, compute_v=True):
    """Compute the bidiagonalization of matrix A"""
    A = np.asarray(A, dtype=np.float64).copy()

    m, n = A.shape

    if m < n:
        raise ValueError(
            f"bidiagonalize requires m >= n, got {m}x{n} (transpose first if wide)"
        )

    left, right = [], []
    d = np.zeros(n)
    f = np.zeros(n - 1)

    for j in range(n):
        v_l, beta_l = householder.house(A[j:, j])

        # Level 2 BLAS Optimization: A = A - beta * v * (v.T * A)
        A[j:, j:] -= beta_l * np.outer(v_l, v_l @ A[j:, j:])

        left.append((j, v_l.copy(), beta_l))
        d[j] = A[j, j]

        if j <= n - 2:
            v_r, beta_r = householder.house(A[j, j + 1 :])

            A[j:, j + 1 :] -= beta_r * np.outer(A[j:, j + 1 :] @ v_r, v_r)
            right.append((j, v_r.copy(), beta_r))
            f[j] = A[j, j + 1]

    U = None
    if compute_u:
        U = np.eye(m, n)
        for j, v, beta in reversed(left):
            U[j:, :] -= beta * np.outer(v, v @ U[j:, :])

    V = None
    if compute_v:
        V = np.eye(n)
        for j, v, beta in reversed(right):
            V[j + 1 :, :] -= beta * np.outer(v, v @ V[j + 1 :, :])

    return U, d, f, V


def bidiag_to_full(d, f):
    """Reconstruct the square (n x n) bidiagonal matrix from d, f."""
    n = len(d)
    B = np.diag(d).astype(np.float64)
    if n > 1:
        B[np.arange(n - 1), np.arange(1, n)] = f
    return B
