"""Experiment: kNN classifier on MNIST (uses KNNClassifier.predict directly).

Sweeps k, reports validation accuracy and prediction time, then final test
accuracy at the best k. Dumps results/knn_sweep.npz for post-processing.

Run from the repo root:
    python experiments/experiment_knn.py

MEMORY NOTE
-----------
KNNClassifier.predict builds the full (M_test, N_train) squared-distance
matrix in one allocation. At full MNIST that is (10000, 60000) float64
~ 4.8 GB. If your machine has the RAM, leave CHUNK_TEST = None (one shot).
If not, set CHUNK_TEST to e.g. 2000: the chunk wrapper calls your unmodified
predict() on slices of the test set, bounding the matrix to
(2000, 60000) ~ 0.96 GB. Either way, YOUR predict does the work.
"""

import time
from pathlib import Path

import numpy as np

from data.mnist import load_mnist
from classifiers.knn import KNNClassifier


K_VALUES = [1, 3, 5, 7]
CHUNK_TEST = None  # None = one-shot
RESULTS_DIR = Path("results")


def predict_maybe_chunked(clf, X, chunk=CHUNK_TEST):
    """Call clf.predict directly, optionally over test-set slices to bound
    memory. The per-slice call is YOUR predict, unchanged."""
    if chunk is None:
        return clf.predict(X)
    out = np.empty(X.shape[0], dtype=clf.y_train.dtype)
    for start in range(0, X.shape[0], chunk):
        end = min(start + chunk, X.shape[0])
        out[start:end] = clf.predict(X[start:end])
    return out


def score(clf, X, y, chunk=CHUNK_TEST):
    return float(np.mean(predict_maybe_chunked(clf, X, chunk) == y))


def main():
    data = load_mnist("data/raw")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    print(
        f"Loaded MNIST: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}\n"
    )

    print("=== Sweep over k (full train, scored on val) ===")
    print(f"{'k':>4} | {'val acc':>9} | {'predict time':>13}")
    print("-" * 36)

    val_accs = []
    predict_times = []
    for k in K_VALUES:
        clf = KNNClassifier(k=k).fit(X_train, y_train)
        t0 = time.perf_counter()
        acc = score(clf, X_val, y_val)
        t_pred = time.perf_counter() - t0
        val_accs.append(acc)
        predict_times.append(t_pred)
        print(f"{k:>4} | {acc:>9.4f} | {t_pred:>12.2f}s")

    best_k = K_VALUES[int(np.argmax(val_accs))]
    print(f"\nBest k on validation: {best_k} (val acc {max(val_accs):.4f})")

    # final test accuracy at best k
    final_clf = KNNClassifier(k=best_k).fit(X_train, y_train)
    t0 = time.perf_counter()
    final_test_acc = score(final_clf, X_test, y_test)
    final_test_time = time.perf_counter() - t0
    print(
        f"FINAL test accuracy at k={best_k}: {final_test_acc:.4f} "
        f"(predict {final_test_time:.1f}s)"
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    np.savez(
        RESULTS_DIR / "knn_sweep.npz",
        k_values=np.array(K_VALUES),
        val_accuracies=np.array(val_accs),
        predict_times=np.array(predict_times),
        best_k=best_k,
        final_test_acc=final_test_acc,
        final_test_predict_time=final_test_time,
    )
    print(f"\nSaved -> {RESULTS_DIR / 'knn_sweep.npz'}")


if __name__ == "__main__":
    main()
