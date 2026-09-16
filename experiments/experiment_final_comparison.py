"""Experiment: unified head-to-head comparison of all five classifiers.

This produces the summary table that is the conclusion of the report: each
method at its best hyperparameter, with test accuracy, fit time, and predict
time side by side. Also persists the SVD full-data and projector-equivalence
numbers (previously one-off scripts) so they live in results/.

Run from the repo root (AFTER the per-method sweeps, so you know best params):
    python experiments/experiment_final_comparison.py

Dumps results/final_comparison.npz.
"""

import time
import warnings
from pathlib import Path

import numpy as np

import sys

# Put the repo root on sys.path so `python experiments/<script>.py` works
# as well as `python -m experiments.<script>`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.mnist import load_mnist
from classifiers.svd_subspace import SVDSubspaceClassifier, numpy_svd_engine
from linalg.svd import svd as scratch_svd
from classifiers.knn import KNNClassifier
from classifiers.mls import MultivariateLSClassifier
from classifiers.sklearn_ref import SklearnKNNClassifier, SklearnRidgeClassifier

RESULTS_DIR = Path("results")

# Best hyperparameters from your sweeps (edit to match YOUR results)
BEST_R = 30
BEST_K = 1
BEST_ALPHA = 600.0


def timed_fit_predict(clf, X_train, y_train, X_test, y_test):
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    preds = clf.predict(X_test)
    t_pred = time.perf_counter() - t0
    acc = float(np.mean(preds == y_test))
    return acc, t_fit, t_pred


def main():
    data = load_mnist("data/raw")
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    print(f"Loaded MNIST: train {X_train.shape}, test {X_test.shape}\n")
    RESULTS_DIR.mkdir(exist_ok=True)

    methods = {
        "SVD-subspace (numpy)": SVDSubspaceClassifier(
            r=BEST_R, svd_engine=numpy_svd_engine
        ),
        "kNN (scratch)": KNNClassifier(k=BEST_K),
        "Least-squares": MultivariateLSClassifier(num_classes=10),
        "kNN (sklearn)": SklearnKNNClassifier(k=BEST_K),
        "Ridge (sklearn)": SklearnRidgeClassifier(alpha=BEST_ALPHA),
    }

    print(f"{'method':<22} | {'test acc':>8} | {'fit (s)':>8} | {'predict (s)':>11}")
    print("-" * 60)
    names, accs, fits, preds = [], [], [], []
    for name, clf in methods.items():
        acc, t_fit, t_pred = timed_fit_predict(clf, X_train, y_train, X_test, y_test)
        names.append(name)
        accs.append(acc)
        fits.append(t_fit)
        preds.append(t_pred)
        print(f"{name:<22} | {acc:>8.4f} | {t_fit:>8.2f} | {t_pred:>11.2f}")

    # --- SVD full-data scratch timing + projector equivalence (persisted) ---
    print("\n=== SVD scratch: full-data timing + projector equivalence ===")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        clf_sc = SVDSubspaceClassifier(r=BEST_R, svd_engine=scratch_svd).fit(
            X_train, y_train
        )
        scratch_full_fit = time.perf_counter() - t0
    clf_np = SVDSubspaceClassifier(r=BEST_R, svd_engine=numpy_svd_engine).fit(
        X_train, y_train
    )
    worst_proj = 0.0
    for Uk_sc, Uk_np in zip(clf_sc.ubases_, clf_np.ubases_):
        worst_proj = max(worst_proj, np.max(np.abs(Uk_sc @ Uk_sc.T - Uk_np @ Uk_np.T)))
    scratch_acc = clf_sc.score(X_test, y_test)
    print(f"  scratch full-data fit: {scratch_full_fit:.1f}s")
    print(f"  scratch test acc:      {scratch_acc:.4f}")
    print(f"  worst projector diff:  {worst_proj:.2e}")

    np.savez(
        RESULTS_DIR / "final_comparison.npz",
        method_names=np.array(names),
        test_accuracies=np.array(accs),
        fit_times=np.array(fits),
        predict_times=np.array(preds),
        best_r=BEST_R,
        best_k=BEST_K,
        best_alpha=BEST_ALPHA,
        svd_scratch_full_fit_time=scratch_full_fit,
        svd_scratch_test_acc=scratch_acc,
        svd_projector_diff=worst_proj,
    )
    print(f"\nSaved -> {RESULTS_DIR / 'final_comparison.npz'}")


if __name__ == "__main__":
    main()
