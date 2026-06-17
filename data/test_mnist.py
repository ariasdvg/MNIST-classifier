import pytest
import numpy as np
from mnist import (
    load_mnist,
    stack_by_class,
)


@pytest.fixture
def data():
    return load_mnist("raw", seed=0)


def test_shapes(data):
    assert data["X_train"].shape == (50_000, 784)
    assert data["X_val"].shape == (10_000, 784)
    assert data["X_test"].shape == (10_000, 784)
    assert data["y_train"].shape == (50_000,)


def test_dtypes_and_ranges(data):
    assert data["X_train"].dtype == np.float32
    assert 0.0 <= data["X_train"].min() and data["X_train"].max() <= 1.0
    assert data["y_train"].min() >= 0 and data["y_train"].max() <= 9


def test_determinism():
    a = load_mnist("raw", seed=0)
    b = load_mnist("raw", seed=0)
    np.testing.assert_array_equal(a["X_train"], b["X_train"])


def test_stack_by_class(data):
    stacks = stack_by_class(data["X_train"], data["y_train"])
    for k in range(10):
        assert stacks[k].shape[0] == 784
        assert stacks[k].shape[1] == np.sum(data["y_train"] == k)
