"""Property-based test suite for the from-scratch SVD pipeline.

Run from the repo root:  python -m pytest tests/test_svd.py -v

Tests are organized bottom-up: primitives (house, givens), then the
bidiagonalization, then the core Golub-Kahan step (against a matrix
reference), then the full svd() wrapper on an edge-case battery.
"""

import numpy as np
import pytest
import warnings

from linalg.householder import house
from linalg.givens import givens
from linalg.bidiagonal import bidiagonalize, bidiag_to_full
from linalg.svd_utils import wilkinson_shift, deflate, find_block, fix_signs
from linalg.svd_core import gk_step
from linalg.svd import svd

RNG = np.random.default_rng(0)


# ============================================================
# Primitives
# ============================================================


@pytest.mark.parametrize("n", [1, 2, 5, 20])
def test_house_zeros_tail(n):
    x = RNG.standard_normal(n)
    v, beta = house(x)
    H = np.eye(n) - beta * np.outer(v, v)
    Hx = H @ x
    assert np.allclose(Hx[1:], 0.0, atol=1e-12)
    assert np.isclose(abs(Hx[0]), np.linalg.norm(x), atol=1e-12)


@pytest.mark.parametrize("n", [2, 5, 20])
def test_house_orthogonal(n):
    x = RNG.standard_normal(n)
    v, beta = house(x)
    H = np.eye(n) - beta * np.outer(v, v)
    assert np.allclose(H.T @ H, np.eye(n), atol=1e-12)


def test_house_aligned():
    v, beta = house(np.array([5.0, 0.0, 0.0]))
    assert beta == 0.0


def test_house_neg_aligned():
    x = np.array([-5.0, 0.0, 0.0])
    v, beta = house(x)
    H = np.eye(3) - beta * np.outer(v, v)
    assert np.allclose(H @ x, [5.0, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize(
    "a,b", [(3.0, 4.0), (1.0, 1e-9), (0.0, 2.0), (-2.0, 5.0), (7.0, -3.0)]
)
def test_givens_zeroes(a, b):
    c, s = givens(a, b)
    assert np.isclose(c**2 + s**2, 1.0, atol=1e-14)
    # Convention: G=[[c,s],[-s,c]];  G.T @ [a;b] zeroes the second entry.
    G = np.array([[c, s], [-s, c]])
    out = G.T @ np.array([a, b])
    assert np.isclose(out[1], 0.0, atol=1e-12)


def test_givens_b_zero():
    c, s = givens(3.0, 0.0)
    assert (c, s) == (1.0, 0.0)


# ============================================================
# Bidiagonalization
# ============================================================


@pytest.mark.parametrize("m,n", [(6, 4), (5, 5), (10, 3), (1, 1), (8, 1), (40, 25)])
def test_bidiag_reconstruction(m, n):
    A = RNG.standard_normal((m, n))
    U, d, f, V = bidiagonalize(A)
    B = bidiag_to_full(d, f)
    assert np.allclose(U @ B @ V.T, A, atol=1e-10)


@pytest.mark.parametrize("m,n", [(6, 4), (5, 5), (10, 3)])
def test_bidiag_orthonormal(m, n):
    A = RNG.standard_normal((m, n))
    U, d, f, V = bidiagonalize(A)
    assert np.allclose(U.T @ U, np.eye(n), atol=1e-10)
    assert np.allclose(V.T @ V, np.eye(n), atol=1e-10)


@pytest.mark.parametrize("m,n", [(6, 4), (5, 5), (10, 3), (40, 25)])
def test_bidiag_preserves_sv(m, n):
    A = RNG.standard_normal((m, n))
    _, d, f, _ = bidiagonalize(A)
    s_A = np.linalg.svd(A, compute_uv=False)
    s_B = np.linalg.svd(bidiag_to_full(d, f), compute_uv=False)
    assert np.allclose(np.sort(s_A)[::-1], np.sort(s_B)[::-1], atol=1e-10)


def test_bidiag_rejects_wide():
    with pytest.raises(ValueError):
        bidiagonalize(RNG.standard_normal((3, 7)))


def test_bidiag_skip_u():
    A = RNG.standard_normal((6, 4))
    U, d, f, V = bidiagonalize(A, compute_u=False)
    assert U is None and V is not None


# ============================================================
# Deflation / block finding / shift
# ============================================================


def test_deflate_zeros_small():
    d = np.array([1.0, 1.0, 1.0])
    f = np.array([1e-20, 0.5])
    deflate(d, f)
    assert f[0] == 0.0 and f[1] == 0.5


def test_find_block_full_diagonal():
    assert find_block(np.array([0.0, 0.0, 0.0]), 4) is None


def test_find_block_trailing():
    # f = [0.5, 0, 0] -> block is rows/cols 0..1
    lo, hi = find_block(np.array([0.5, 0.0, 0.0]), 4)
    assert (lo, hi) == (0, 1)


def test_wilkinson_matches_eig():
    d = np.array([2.0, 3.0, 4.0, 5.0])
    f = np.array([1.0, 1.0, 1.0])
    mu = wilkinson_shift(d, f, 0, 3)
    a = d[2] ** 2 + f[1] ** 2
    b = d[2] * f[2]
    c = d[3] ** 2 + f[2] ** 2
    T = np.array([[a, b], [b, c]])
    eigs = np.linalg.eigvalsh(T)
    # mu is the eigenvalue closer to c
    closer = eigs[np.argmin(np.abs(eigs - c))]
    assert np.isclose(mu, closer, atol=1e-10)


# ============================================================
# Golub-Kahan step: against a matrix reference
# ============================================================


def _gk_reference(d, f, lo, hi):
    """Trusted GK step that materializes B and applies matrix rotations."""
    n = len(d)
    B = bidiag_to_full(d, f)

    def gmat(i, j, c, s):
        G = np.eye(n)
        G[i, i] = c
        G[j, j] = c
        G[i, j] = s
        G[j, i] = -s
        return G

    mu = wilkinson_shift(d, f, lo, hi)
    y = d[lo] ** 2 - mu
    z = d[lo] * f[lo]
    for k in range(lo, hi):
        c, s = givens(y, z)
        B = B @ gmat(k, k + 1, c, s)
        y = B[k, k]
        z = B[k + 1, k]
        c, s = givens(y, z)
        B = gmat(k, k + 1, c, s).T @ B
        if k < hi - 1:
            y = B[k, k + 1]
            z = B[k, k + 2]
    return np.diag(B).copy(), np.diag(B, 1).copy()


@pytest.mark.parametrize("seed", range(50))
def test_gk_step_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 9))
    d = rng.standard_normal(n)
    f = rng.standard_normal(n - 1)
    dr, fr = _gk_reference(d.copy(), f.copy(), 0, n - 1)
    db, fb = d.copy(), f.copy()
    gk_step(db, fb, None, None, 0, n - 1)
    assert np.allclose(db, dr, atol=1e-9)
    assert np.allclose(fb, fr, atol=1e-9)


@pytest.mark.parametrize("seed", range(20))
def test_gk_step_preserves_frobenius(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 8))
    d = rng.standard_normal(n)
    f = rng.standard_normal(n - 1)
    before = np.sum(d**2) + np.sum(f**2)
    gk_step(d, f, None, None, 0, n - 1)
    after = np.sum(d**2) + np.sum(f**2)
    assert np.isclose(before, after, atol=1e-9)


# ============================================================
# Full svd() wrapper
# ============================================================


def _check_full(A, atol=1e-7):
    U, s, V = svd(A, want="both")
    s_np = np.linalg.svd(A, compute_uv=False)
    n = min(A.shape)
    assert np.allclose(np.sort(s)[::-1][:n], np.sort(s_np)[::-1][:n], atol=atol)
    r = len(s)
    recon = (U[:, :r] * s[:r]) @ V[:, :r].T
    assert np.allclose(recon, A, atol=atol)
    assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-8)
    assert np.allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-8)


@pytest.mark.parametrize(
    "m,n", [(20, 10), (10, 10), (100, 30), (2, 2), (1, 1), (15, 4)]
)
def test_svd_random(m, n):
    _check_full(RNG.standard_normal((m, n)))


@pytest.mark.parametrize("m,n", [(6, 15), (4, 20), (3, 8)])
def test_svd_wide(m, n):
    _check_full(RNG.standard_normal((m, n)))


def test_svd_repeated_sv():
    Q1, _ = np.linalg.qr(RNG.standard_normal((5, 5)))
    Q2, _ = np.linalg.qr(RNG.standard_normal((5, 5)))
    A = Q1 @ np.diag([3.0, 3.0, 3.0, 1.0, 0.5]) @ Q2.T
    _check_full(A)


def test_svd_tight_cluster():
    Q1, _ = np.linalg.qr(RNG.standard_normal((5, 5)))
    Q2, _ = np.linalg.qr(RNG.standard_normal((5, 5)))
    A = Q1 @ np.diag([1.0, 1.0 + 1e-9, 0.5, 0.25, 0.1]) @ Q2.T
    _check_full(A)


def test_svd_rank_deficient():
    # Triggers the zero-diagonal fallback; just check singular values.
    Q1, _ = np.linalg.qr(RNG.standard_normal((5, 5)))
    Q2, _ = np.linalg.qr(RNG.standard_normal((5, 5)))
    A = Q1 @ np.diag([2.0, 1.0, 0.0, 0.0, 0.0]) @ Q2.T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        U, s, V = svd(A, want="both")
    s_np = np.linalg.svd(A, compute_uv=False)
    assert np.allclose(np.sort(s)[::-1], np.sort(s_np)[::-1], atol=1e-7)


def test_svd_diagonal():
    _check_full(np.diag([4.0, 3.0, 2.0, 1.0]))


def test_svd_want_v_wide():
    A = RNG.standard_normal((6, 20))
    V, s = svd(A, want="V")
    s_np = np.linalg.svd(A, compute_uv=False)
    assert np.allclose(np.sort(s)[::-1][:6], np.sort(s_np)[::-1][:6], atol=1e-7)
    assert V.shape[0] == 20


def test_svd_truncation():
    A = RNG.standard_normal((30, 20))
    U, s, V = svd(A, want="both", k=5)
    assert len(s) == 5 and U.shape[1] == 5 and V.shape[1] == 5
    s_np = np.linalg.svd(A, compute_uv=False)
    assert np.allclose(s, np.sort(s_np)[::-1][:5], atol=1e-7)


def test_svd_singular_values_descending():
    A = RNG.standard_normal((12, 8))
    _, s, _ = svd(A, want="both")
    assert np.all(np.diff(s) <= 1e-12)
    assert np.all(s >= -1e-12)
