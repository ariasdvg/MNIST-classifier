import struct
from pathlib import Path

import numpy as np


# low level IDX readers


def _read_idx_images(filepath: Path) -> np.ndarray:
    """Read an IDX3-ubyte image file. Returns an (N, 28, 28) uint8 array."""
    with open(filepath, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic in {filepath}: expected 2051, got {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def _read_idx_labels(filepath: Path) -> np.ndarray:
    """Read an IDX1-ubyte label file. Returns an (N,) uint8 array"""
    with open(filepath, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic in {filepath}: expected 2049, got {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.shape[0] != n:
        raise ValueError(f"Expected {n} labels in {filepath}, got {data.shape[0]}")
    return data


# load MNIST


def load_mnist(
    data_dir: str | Path,
    validation_size: int = 10_000,
    seed: int = 0,
) -> dict[str, np.ndarray]:

    data_dir = Path(data_dir)
    train_images = _read_idx_images(data_dir / "train-images.idx3-ubyte")
    train_labels = _read_idx_labels(data_dir / "train-labels.idx1-ubyte")
    test_images = _read_idx_images(data_dir / "t10k-images.idx3-ubyte")
    test_labels = _read_idx_labels(data_dir / "t10k-labels.idx1-ubyte")

    X_train_full = train_images.reshape(-1, 784).astype(np.float64) / 255.0
    X_test = test_images.reshape(-1, 784).astype(np.float64) / 255.0

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X_train_full.shape[0])
    val_idx = perm[:validation_size]
    train_idx = perm[validation_size:]

    return {
        "X_train": X_train_full[train_idx],
        "y_train": train_labels[train_idx],
        "X_val": X_train_full[val_idx],
        "y_val": train_labels[val_idx],
        "X_test": X_test,
        "y_test": test_labels,
    }


def stack_by_class(X: np.ndarray, y: np.ndarray) -> dict[int, np.ndarray]:
    return {k: X[y == k].T for k in range(10)}
