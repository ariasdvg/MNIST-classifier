"""Pytest suite for SVDSubspaceClassifier, covering BOTH SVD engines.

PASS/FAIL gates (not an experiment). Fast: small synthetic data plus an
optional real-MNIST subsample that auto-skips if the data isn't present.

The key idea: every behavioral test is parametrized over both engines
(scratch and numpy), so a regression in either is caught. A dedicated
equivalence test asserts the two engines learn the same subspaces.

Run from the repo root:  python -m pytest tests/test_svd_subspace.py -v
"""

from pathlib import Path

import numpy as np
import pytest

from classifiers.svd_subspace import (
    SVDSubspaceClassifier,
    numpy_svd_engine,
)
from linalg.svd import svd as scratch_svd


# Both engines, referenced by name so test output is readable.
ENGINES = [
    pytest.param(scratch_svd, id="scratch"),
    pytest.param(numpy_svd_engine, id="numpy"),
]


# ---------- synthetic data ----------


def make_subspace_data(n_per=100, noise=0.4, dim=64, sub=3, seed=0):
    """10 classes, each a `sub`-dimensional subspace + noise, in R^dim."""
    rng = np.random.default_rng(seed)
    bases = [
        np.random.default_rng(100 + k).standard_normal((dim, sub)) for k in range(10)
    ]
    X, y = [], []
    for k in range(10):
        for _ in range(n_per):
            X.append(
                bases[k] @ rng.standard_normal(sub) + noise * rng.standard_normal(dim)
            )
            y.append(k)
    X = np.array(X)
    y = np.array(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ---------- structural tests (run for BOTH engines) ----------


@pytest.mark.parametrize("engine", ENGINES)
def test_fit_returns_self(engine):
    X, y = make_subspace_data(n_per=20)
    clf = SVDSubspaceClassifier(r=3, svd_engine=engine)
    assert clf.fit(X, y) is clf


@pytest.mark.parametrize("engine", ENGINES)
def test_fit_stores_ten_bases(engine):
    X, y = make_subspace_data(n_per=20)
    clf = SVDSubspaceClassifier(r=3, svd_engine=engine).fit(X, y)
    assert len(clf.ubases_) == 10


@pytest.mark.parametrize("engine", ENGINES)
def test_bases_have_correct_shape(engine):
    X, y = make_subspace_data(n_per=20, dim=64)
    r = 3
    clf = SVDSubspaceClassifier(r=r, svd_engine=engine).fit(X, y)
    for U_k in clf.ubases_:
        assert U_k.shape[0] == 64  # ambient (pixel) dimension
        assert U_k.shape[1] == r  # r basis vectors


@pytest.mark.parametrize("engine", ENGINES)
def test_bases_orthonormal(engine):
    X, y = make_subspace_data(n_per=30)
    clf = SVDSubspaceClassifier(r=4, svd_engine=engine).fit(X, y)
    for U_k in clf.ubases_:
        gram = U_k.T @ U_k
        assert np.allclose(gram, np.eye(U_k.shape[1]), atol=1e-8)


@pytest.mark.parametrize("engine", ENGINES)
def test_predict_shape_and_range(engine):
    X, y = make_subspace_data(n_per=30)
    clf = SVDSubspaceClassifier(r=3, svd_engine=engine).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (X.shape[0],)
    assert preds.min() >= 0 and preds.max() <= 9


@pytest.mark.parametrize("engine", ENGINES)
def test_refit_does_not_accumulate(engine):
    X, y = make_subspace_data(n_per=20)
    clf = SVDSubspaceClassifier(r=3, svd_engine=engine)
    clf.fit(X, y)
    clf.fit(X, y)
    assert len(clf.ubases_) == 10


@pytest.mark.parametrize("engine", ENGINES)
def test_guard_r_larger_than_samples(engine):
    X, y = make_subspace_data(n_per=3, dim=64)  # only 3 per class
    clf = SVDSubspaceClassifier(r=10, svd_engine=engine).fit(X, y)
    assert clf.predict(X).shape == (X.shape[0],)


# ---------- accuracy tests (run for BOTH engines) ----------


@pytest.mark.parametrize("engine", ENGINES)
def test_recovers_true_dimension(engine):
    X, y = make_subspace_data(n_per=150, noise=0.4, sub=3)
    clf = SVDSubspaceClassifier(r=3, svd_engine=engine).fit(X, y)
    assert clf.score(X, y) > 0.95


@pytest.mark.parametrize("engine", ENGINES)
def test_beats_random_baseline(engine):
    X, y = make_subspace_data(n_per=100, noise=0.6)
    clf = SVDSubspaceClassifier(r=5, svd_engine=engine).fit(X, y)
    assert clf.score(X, y) > 0.5


# ---------- the equivalence test: the two engines must agree ----------


def test_engines_learn_same_subspaces():
    """The whole point of the dual-engine design: scratch and numpy must
    produce the SAME learned subspaces (hence the same predictions)."""
    X, y = make_subspace_data(n_per=80, noise=0.5, dim=80, sub=4, seed=3)

    clf_sc = SVDSubspaceClassifier(r=6, svd_engine=scratch_svd).fit(X, y)
    clf_np = SVDSubspaceClassifier(r=6, svd_engine=numpy_svd_engine).fit(X, y)

    # Compare projectors U_k U_k^T (invariant to column sign and ordering of
    # equal singular values, unlike comparing U_k directly).
    for Uk_sc, Uk_np in zip(clf_sc.ubases_, clf_np.ubases_):
        P_sc = Uk_sc @ Uk_sc.T
        P_np = Uk_np @ Uk_np.T
        assert np.allclose(P_sc, P_np, atol=1e-7)

    # And identical predictions.
    assert np.array_equal(clf_sc.predict(X), clf_np.predict(X))


# ---------- optional real-MNIST subsample (auto-skips if absent) ----------


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _try_load_mnist_subsample(per_class=200, n_test=1000):
    try:
        from data.mnist import load_mnist

        data = load_mnist(RAW_DIR)
    except Exception:
        return None
    rng = np.random.default_rng(0)
    Xtr, ytr = data["X_train"], data["y_train"]
    idx = []
    for k in range(10):
        members = np.where(ytr == k)[0]
        idx.extend(rng.choice(members, per_class, replace=False))
    idx = np.array(idx)
    return Xtr[idx], ytr[idx], data["X_test"][:n_test], data["y_test"][:n_test]


@pytest.mark.parametrize("engine", ENGINES)
def test_real_mnist_subsample_accuracy(engine):
    bundle = _try_load_mnist_subsample()
    if bundle is None:
        pytest.skip(f"MNIST not available at {RAW_DIR}")
    Xtr, ytr, Xte, yte = bundle
    clf = SVDSubspaceClassifier(r=10, svd_engine=engine).fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    assert acc > 0.85, f"real-MNIST subsample accuracy too low: {acc}"
