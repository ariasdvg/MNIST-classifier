import time

import sys
from pathlib import Path

# Put the repo root on sys.path so `python experiments/<script>.py` works
# as well as `python -m experiments.<script>`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.mnist import load_mnist
from classifiers.svd_subspace import SVDSubspaceClassifier, numpy_svd_engine


def main():
    data = load_mnist("data/raw")
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    print("=" * 48)
    print("  SVD-subspace classifier — full MNIST")
    print("=" * 48)
    print(f"  train: {X_train.shape[0]:>6} images")
    print(f"  test:   {X_test.shape[0]:>6} images")
    print(f"  r = 30   (engine: from-scratch SVD)")
    print("-" * 48)

    clf = SVDSubspaceClassifier(r=30, svd_engine=numpy_svd_engine)

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    acc = clf.score(X_test, y_test)
    t_pred = time.perf_counter() - t0

    print(f"  fit time:        {t_fit:8.2f}s")
    print(f"  predict time:    {t_pred:8.2f}s")
    print(f"  validation acc:  {acc:8.4f}")
    print("=" * 48)


if __name__ == "__main__":
    main()
