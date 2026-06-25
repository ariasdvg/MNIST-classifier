"""Experiment: multivariate least-squares classifier on MNIST.

Plain LS has no hyperparameter to sweep, so this:
  1. runs the from-scratch LS, reports val + test accuracy and timing,
  2. compares against sklearn Ridge (the regressor) at small alpha as the
     golden reference (should agree closely), and
  3. sweeps alpha on the Ridge reference to show the effect of regularization.

Run from the repo root:
    python experiments/experiment_ls.py

Dumps results/ls_sweep.npz.
"""

import time
from pathlib import Path

import numpy as np

from data.mnist import load_mnist
from classifiers.mls import MultivariateLSClassifier
from classifiers.sklearn_ref import SklearnRidgeClassifier


ALPHA_VALUES = [0.001, 0.1, 1.0, 10.0, 100.0]
RESULTS_DIR = Path("results")


def main():
    data = load_mnist("data/raw")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    print(
        f"Loaded MNIST: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}\n"
    )

    # --- from-scratch LS ---
    print("=== From-scratch multivariate LS ===")
    t0 = time.perf_counter()
    clf = MultivariateLSClassifier(num_classes=10).fit(X_train, y_train)
    t_fit = time.perf_counter() - t0
    ls_val = clf.score(X_val, y_val)
    ls_test = clf.score(X_test, y_test)
    print(f"  fit time:        {t_fit:8.2f}s")
    print(f"  validation acc:  {ls_val:8.4f}")
    print(f"  test acc:        {ls_test:8.4f}\n")

    # --- golden reference: sklearn Ridge, sweep alpha ---
    print("=== sklearn Ridge reference (sweep alpha, scored on val) ===")
    print(f"{'alpha':>8} | {'val acc':>9} | {'agree w/ LS':>11}")
    print("-" * 36)
    ref_val_accs = []
    ref_agreements = []
    ls_val_preds = clf.predict(X_val)
    for alpha in ALPHA_VALUES:
        ref = SklearnRidgeClassifier(alpha=alpha).fit(X_train, y_train)
        ref_preds = ref.predict(X_val)
        acc = float(np.mean(ref_preds == y_val))
        agree = float(np.mean(ref_preds == ls_val_preds))
        ref_val_accs.append(acc)
        ref_agreements.append(agree)
        print(f"{alpha:>8.3f} | {acc:>9.4f} | {agree:>11.4f}")

    best_alpha = ALPHA_VALUES[int(np.argmax(ref_val_accs))]
    ref_best = SklearnRidgeClassifier(alpha=best_alpha).fit(X_train, y_train)
    ref_test = ref_best.score(X_test, y_test)
    print(f"\nBest Ridge alpha on val: {best_alpha} (val acc {max(ref_val_accs):.4f})")
    print(f"Ridge test acc at best alpha: {ref_test:.4f}")
    print(f"\nOur LS test: {ls_test:.4f}   |   Ridge test: {ref_test:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    np.savez(
        RESULTS_DIR / "ls_sweep.npz",
        ls_val_acc=ls_val,
        ls_test_acc=ls_test,
        ls_fit_time=t_fit,
        alpha_values=np.array(ALPHA_VALUES),
        ref_val_accuracies=np.array(ref_val_accs),
        ref_agreements=np.array(ref_agreements),
        best_alpha=best_alpha,
        ref_test_acc=ref_test,
    )
    print(f"\nSaved -> {RESULTS_DIR / 'ls_sweep.npz'}")


if __name__ == "__main__":
    main()
