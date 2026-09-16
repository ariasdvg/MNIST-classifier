"""Experiment: scikit-learn RidgeClassifier on MNIST (library reference).

Golden reference for your from-scratch LeastSquaresClassifier. Sweeps the
regularization strength alpha (the library's lambda), scores on validation,
reports final test accuracy at the best alpha, and dumps
results/sklearn_ridge_sweep.npz.

Run from the repo root:
    python experiments/experiment_sklearn_ridge.py
"""

import time
from pathlib import Path

import numpy as np

import sys

# Put the repo root on sys.path so `python experiments/<script>.py` works
# as well as `python -m experiments.<script>`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.mnist import load_mnist
from classifiers.sklearn_ref import SklearnRidgeClassifier


# alpha is ridge regularization strength (lambda). alpha -> 0 approaches
# ordinary least squares; larger alpha = more regularization.
ALPHA_VALUES = [
    0.0,
    0.1,
    1.0,
    10.0,
    100.0,
    150.0,
    200.0,
    250.0,
    350.0,
    500.0,
    600.0,
    700.0,
    750.0,
    800.0,
    900.0,
    1000.0,
]
RESULTS_DIR = Path("results")


def main():
    data = load_mnist("data/raw")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    print(
        f"Loaded MNIST: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}\n"
    )

    print("=== Sweep over alpha (sklearn Ridge, full train, scored on val) ===")
    print(f"{'alpha':>8} | {'val acc':>9} | {'fit time':>9}")
    print("-" * 34)

    val_accs = []
    fit_times = []
    for alpha in ALPHA_VALUES:
        clf = SklearnRidgeClassifier(alpha=alpha)

        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        t_fit = time.perf_counter() - t0
        acc = clf.score(X_val, y_val)

        val_accs.append(acc)
        fit_times.append(t_fit)
        print(f"{alpha:>8.1f} | {acc:>9.4f} | {t_fit:>8.2f}s")

    best_alpha = ALPHA_VALUES[int(np.argmax(val_accs))]
    print(f"\nBest alpha on validation: {best_alpha} (val acc {max(val_accs):.4f})")

    final_clf = SklearnRidgeClassifier(alpha=best_alpha).fit(X_train, y_train)
    final_test_acc = final_clf.score(X_test, y_test)
    print(f"FINAL test accuracy at alpha={best_alpha}: {final_test_acc:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    np.savez(
        RESULTS_DIR / "sklearn_ridge_sweep.npz",
        alpha_values=np.array(ALPHA_VALUES),
        val_accuracies=np.array(val_accs),
        fit_times=np.array(fit_times),
        best_alpha=best_alpha,
        final_test_acc=final_test_acc,
    )
    print(f"\nSaved -> {RESULTS_DIR / 'sklearn_ridge_sweep.npz'}")


if __name__ == "__main__":
    main()
