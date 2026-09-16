import numpy as np
import pytest

from linalg.householder import house
from linalg.givens import givens
from linalg.bidiagonal import bidiagonalize, bidiag_to_full

RNG = np.random.default_rng(0)


# ---------- house ----------


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


def test_house_already_aligned():
    v, beta = house(np.array([5.0, 0.0, 0.0]))
    assert beta == 0.0  # no reflection needed


def test_house_negatively_aligned():
    x = np.array([-5.0, 0.0, 0.0])
    v, beta = house(x)
    H = np.eye(3) - beta * np.outer(v, v)
    assert np.allclose(H @ x, [5.0, 0.0, 0.0], atol=1e-12)


# ---------- givens ----------


@pytest.mark.parametrize(
    "a,b", [(3.0, 4.0), (1.0, 1e-9), (0.0, 2.0), (-2.0, 5.0), (7.0, -3.0)]
)
def test_givens_zeroes_b(a, b):
    c, s = givens(a, b)
    assert np.isclose(c**2 + s**2, 1.0, atol=1e-14)
    assert np.isclose(s * a + c * b, 0.0, atol=1e-12)  # zeroing condition


def test_givens_b_zero():
    c, s = givens(3.0, 0.0)
    assert (c, s) == (1.0, 0.0)


# ---------- bidiagonalize ----------


@pytest.mark.parametrize("m,n", [(6, 4), (5, 5), (10, 3), (1, 1), (8, 1)])
def test_bidiag_reconstruction(m, n):
    A = RNG.standard_normal((m, n))
    U, d, f, V = bidiagonalize(A)
    B = bidiag_to_full(d, f)
    assert np.allclose(U @ B @ V.T, A, atol=1e-10)


@pytest.mark.parametrize("m,n", [(6, 4), (5, 5), (10, 3)])
def test_bidiag_factors_orthonormal(m, n):
    A = RNG.standard_normal((m, n))
    U, d, f, V = bidiagonalize(A)
    assert np.allclose(U.T @ U, np.eye(n), atol=1e-10)
    assert np.allclose(V.T @ V, np.eye(n), atol=1e-10)


@pytest.mark.parametrize("m,n", [(6, 4), (5, 5), (10, 3), (40, 25)])
def test_bidiag_preserves_singular_values(m, n):
    # Convention-independent: bidiagonalization is orthogonal equivalence,
    # so sigma(B) == sigma(A) even if U/V accumulation had a sign bug.
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
    _, d2, f2, _ = bidiagonalize(A, compute_u=True)
    assert np.allclose(d, d2) and np.allclose(f, f2)
