"""Experiment: scikit-learn KNeighborsClassifier on MNIST (library reference).

Mirrors experiment_knn.py so the from-scratch and library kNN can be put on
the same axes. Sweeps k, scores on validation, reports final test accuracy at
the best k, and dumps results/sklearn_knn_sweep.npz.

Run from the repo root:
    python experiments/experiment_sklearn_knn.py
"""

import time
from pathlib import Path

import numpy as np

from data.mnist import load_mnist
from classifiers.sklearn_ref import SklearnKNNClassifier


K_VALUES = [1, 3, 5, 7]
RESULTS_DIR = Path("results")


def main():
    data = load_mnist("data/raw")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    print(
        f"Loaded MNIST: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}\n"
    )

    print("=== Sweep over k (sklearn kNN, full train, scored on val) ===")
    print(f"{'k':>4} | {'val acc':>9} | {'fit time':>9} | {'predict time':>13}")
    print("-" * 48)

    val_accs = []
    fit_times = []
    predict_times = []
    for k in K_VALUES:
        clf = SklearnKNNClassifier(k=k)

        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        t_fit = time.perf_counter() - t0

        t0 = time.perf_counter()
        acc = clf.score(X_val, y_val)
        t_pred = time.perf_counter() - t0

        val_accs.append(acc)
        fit_times.append(t_fit)
        predict_times.append(t_pred)
        print(f"{k:>4} | {acc:>9.4f} | {t_fit:>8.2f}s | {t_pred:>12.2f}s")

    best_k = K_VALUES[int(np.argmax(val_accs))]
    print(f"\nBest k on validation: {best_k} (val acc {max(val_accs):.4f})")

    final_clf = SklearnKNNClassifier(k=best_k).fit(X_train, y_train)
    t0 = time.perf_counter()
    final_test_acc = final_clf.score(X_test, y_test)
    final_test_time = time.perf_counter() - t0
    print(
        f"FINAL test accuracy at k={best_k}: {final_test_acc:.4f} "
        f"(predict {final_test_time:.1f}s)"
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    np.savez(
        RESULTS_DIR / "sklearn_knn_sweep.npz",
        k_values=np.array(K_VALUES),
        val_accuracies=np.array(val_accs),
        fit_times=np.array(fit_times),
        predict_times=np.array(predict_times),
        best_k=best_k,
        final_test_acc=final_test_acc,
        final_test_predict_time=final_test_time,
    )
    print(f"\nSaved -> {RESULTS_DIR / 'sklearn_knn_sweep.npz'}")


if __name__ == "__main__":
    main()
