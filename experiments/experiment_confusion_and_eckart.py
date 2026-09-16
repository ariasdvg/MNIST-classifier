"""Experiment: confusion matrices for all classifiers + Eckart-Young residual
curve for the SVD-subspace method.

Fills two gaps the per-method sweeps don't cover:
  1. Confusion matrices (promised in preview section 3.7, for every method).
  2. The Eckart-Young relative residual ||A_k - A_{k,r}||_F / ||A_k||_F per
     class and per r (preview 3.7), which empirically validates why the SVD
     subspace method works.

Run from the repo root:
    python experiments/experiment_confusion_and_eckart.py

Dumps results/confusion_matrices.npz and results/eckart_young.npz.
"""

import numpy as np
from pathlib import Path

import sys

# Put the repo root on sys.path so `python experiments/<script>.py` works
# as well as `python -m experiments.<script>`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.mnist import load_mnist, stack_by_class
from classifiers.svd_subspace import SVDSubspaceClassifier, numpy_svd_engine
from classifiers.knn import KNNClassifier
from classifiers.mls import MultivariateLSClassifier
from classifiers.sklearn_ref import SklearnKNNClassifier, SklearnRidgeClassifier

RESULTS_DIR = Path("results")

# Best hyperparameters chosen from the sweeps (edit to match YOUR sweep results)
BEST_R = 30
BEST_K = 1
BEST_ALPHA = 600.0


def confusion_matrix(y_true, y_pred, num_classes=10):
    """(num_classes, num_classes) matrix: entry (i, j) = count of true class i
    predicted as class j. Diagonal = correct."""
    C = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1
    return C


def eckart_young_residuals(X_train, y_train, r_values):
    """For each class k and each r, the relative Frobenius residual of the
    rank-r truncation: ||A_k - A_{k,r}||_F / ||A_k||_F.

    By Eckart-Young this equals sqrt(sum_{i>r} sigma_i^2 / sum_i sigma_i^2),
    so we compute it directly from the singular values (cheap, no need to
    reconstruct A_{k,r})."""
    stacks = stack_by_class(X_train, y_train)
    residuals = np.zeros((10, len(r_values)))
    for k in range(10):
        A_k = stacks[k]  # (784, n_k)
        s = np.linalg.svd(A_k, compute_uv=False)  # singular values desc
        total = np.sum(s**2)
        for j, r in enumerate(r_values):
            tail = np.sum(s[r:] ** 2)  # discarded energy
            residuals[k, j] = np.sqrt(tail / total)
    return residuals


def main():
    data = load_mnist("data/raw")
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    print(f"Loaded MNIST: train {X_train.shape}, test {X_test.shape}\n")

    RESULTS_DIR.mkdir(exist_ok=True)

    # ---- Confusion matrices for all classifiers (at best hyperparameters) ----
    print("=== Computing confusion matrices (test set) ===")
    classifiers = {
        "svd_subspace": SVDSubspaceClassifier(r=BEST_R, svd_engine=numpy_svd_engine),
        "knn": KNNClassifier(k=BEST_K),
        "ls": MultivariateLSClassifier(num_classes=10),
        "sklearn_knn": SklearnKNNClassifier(k=BEST_K),
        "sklearn_ridge": SklearnRidgeClassifier(alpha=BEST_ALPHA),
    }

    confusions = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        C = confusion_matrix(y_test, preds)
        acc = np.trace(C) / C.sum()
        confusions[name] = C
        # most-confused off-diagonal pair
        Coff = C.copy()
        np.fill_diagonal(Coff, 0)
        i, j = np.unravel_index(np.argmax(Coff), Coff.shape)
        print(
            f"  {name:15} acc={acc:.4f}  most confused: "
            f"true {i} -> pred {j} ({Coff[i, j]} times)"
        )

    np.savez(
        RESULTS_DIR / "confusion_matrices.npz",
        **{name: C for name, C in confusions.items()},
    )
    print(f"Saved -> {RESULTS_DIR / 'confusion_matrices.npz'}\n")

    # ---- Eckart-Young residual curve (SVD-subspace justification) ----
    print("=== Computing Eckart-Young relative residuals ===")
    r_values = np.array([1, 2, 5, 10, 15, 20, 30, 50, 100])
    residuals = eckart_young_residuals(X_train, y_train, r_values)
    print(f"  {'r':>4} | " + " ".join(f"d{k}" for k in range(10)))
    for j, r in enumerate(r_values):
        row = " ".join(f"{residuals[k, j]:.2f}" for k in range(10))
        print(f"  {r:>4} | {row}")
    np.savez(RESULTS_DIR / "eckart_young.npz", r_values=r_values, residuals=residuals)
    print(f"Saved -> {RESULTS_DIR / 'eckart_young.npz'}")


if __name__ == "__main__":
    main()
