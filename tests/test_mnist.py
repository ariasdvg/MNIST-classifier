from pathlib import Path

import numpy as np
import pytest

from data.mnist import load_mnist, stack_by_class

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

needs_mnist = pytest.mark.skipif(
    not (RAW_DIR / "train-images.idx3-ubyte").exists(),
    reason=f"MNIST not available at {RAW_DIR}",
)


@pytest.fixture
def data():
    return load_mnist(RAW_DIR, seed=0)


@needs_mnist
def test_shapes(data):
    assert data["X_train"].shape == (50_000, 784)
    assert data["X_val"].shape == (10_000, 784)
    assert data["X_test"].shape == (10_000, 784)
    assert data["y_train"].shape == (50_000,)


@needs_mnist
def test_dtypes_and_ranges(data):
    assert data["X_train"].dtype == np.float64
    assert 0.0 <= data["X_train"].min() and data["X_train"].max() <= 1.0
    assert data["y_train"].min() >= 0 and data["y_train"].max() <= 9


@needs_mnist
def test_determinism():
    a = load_mnist(RAW_DIR, seed=0)
    b = load_mnist(RAW_DIR, seed=0)
    np.testing.assert_array_equal(a["X_train"], b["X_train"])


@needs_mnist
def test_stack_by_class(data):
    stacks = stack_by_class(data["X_train"], data["y_train"])
    for k in range(10):
        assert stacks[k].shape[0] == 784
        assert stacks[k].shape[1] == np.sum(data["y_train"] == k)


def test_stack_by_class_order():
    y = np.array([7, 2, 0, 9, 1, 3])
    X = np.arange(6 * 4).reshape(6, 4).astype(float)
    d = stack_by_class(X, y)
    assert list(d.keys()) == list(range(10))  # all ten, in order
    # column count per class matches label counts
    for k in range(10):
        assert d[k].shape[1] == np.sum(y == k)
