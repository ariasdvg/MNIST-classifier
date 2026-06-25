"""Experiment: SVD-subspace classifier on MNIST, comparing the from-scratch
SVD against numpy.linalg.svd.

This is an EXPERIMENT SCRIPT (not a test): it prints accuracies, sweeps r,
contrasts the two SVD engines, and dumps results to results/ for the
post-processing notebooks to plot.

Run from the repo root:
    python experiments/experiment_svd_subspace.py

What it does
------------
1. Loads MNIST (train / validation / test).
2. Sweeps r with the FAST numpy engine on the full training set, on the
   validation split, to find the best r. (Your from-scratch SVD is O(m n^2)
   and too slow for ten full 60k-class SVDs per r.)
3. Re-runs the chosen best r on a SUBSAMPLE with BOTH engines, to confirm the
   from-scratch SVD reproduces numpy's accuracy. This is the "custom vs
   library" comparison your report needs.
4. Reports the final test accuracy once, at the chosen r.
5. Saves everything to results/svd_subspace_sweep.npz.
"""

import time
from pathlib import Path

import numpy as np

from data.mnist import load_mnist
from classifiers.svd_subspace import (
    SVDSubspaceClassifier,
    numpy_svd_engine,
)
from linalg.svd import svd as scratch_svd


R_VALUES = [1, 2, 5, 10, 15, 20, 30, 50]
SUBSAMPLE_PER_CLASS = 400
RESULTS_DIR = Path("results")


def subsample(X, y, per_class, seed=0):
    """Take `per_class` random examples of each digit (for the slow engine)."""
    rng = np.random.default_rng(seed)
    idx = []
    for k in range(10):
        members = np.where(y == k)[0]
        idx.extend(rng.choice(members, per_class, replace=False))
    idx = np.array(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def sweep_r_numpy(X_train, y_train, X_val, y_val):
    """Fast sweep over r using the numpy engine, scored on validation."""
    print("=== Sweep over r (numpy engine, full train, scored on val) ===")
    print(f"{'r':>4} | {'val acc':>9} | {'fit time':>9}")
    print("-" * 32)
    accs, times = [], []
    for r in R_VALUES:
        t0 = time.perf_counter()
        clf = SVDSubspaceClassifier(r=r, svd_engine=numpy_svd_engine)
        clf.fit(X_train, y_train)
        t_fit = time.perf_counter() - t0
        acc = clf.score(X_val, y_val)
        accs.append(acc)
        times.append(t_fit)
        print(f"{r:>4} | {acc:>9.4f} | {t_fit:>8.2f}s")
    return np.array(accs), np.array(times)


def compare_engines(X_sub, y_sub, X_test, y_test, r):
    """Run BOTH engines at a fixed r on the subsample; confirm they agree."""
    print(
        f"\n=== Scratch vs numpy at r={r} (subsample {SUBSAMPLE_PER_CLASS}/class) ==="
    )

    t0 = time.perf_counter()
    clf_np = SVDSubspaceClassifier(r=r, svd_engine=numpy_svd_engine)
    clf_np.fit(X_sub, y_sub)
    t_np = time.perf_counter() - t0
    acc_np = clf_np.score(X_test, y_test)

    t0 = time.perf_counter()
    clf_sc = SVDSubspaceClassifier(r=r, svd_engine=scratch_svd)
    clf_sc.fit(X_sub, y_sub)
    t_sc = time.perf_counter() - t0
    acc_sc = clf_sc.score(X_test, y_test)

    # also confirm the LEARNED SUBSPACES match (not just the accuracy):
    # compare projectors U_k U_k^T, which are sign/order invariant.
    max_proj_diff = 0.0
    for Uk_np, Uk_sc in zip(clf_np.ubases_, clf_sc.ubases_):
        P_np = Uk_np @ Uk_np.T
        P_sc = Uk_sc @ Uk_sc.T
        max_proj_diff = max(max_proj_diff, np.max(np.abs(P_np - P_sc)))

    print(f"  numpy   : acc={acc_np:.4f}  fit={t_np:6.2f}s")
    print(f"  scratch : acc={acc_sc:.4f}  fit={t_sc:6.2f}s")
    print(f"  |acc diff|            = {abs(acc_np - acc_sc):.2e}")
    print(
        f"  max |P_np - P_scratch| = {max_proj_diff:.2e}  "
        f"(subspaces identical if ~1e-8)"
    )
    print(f"  speed ratio (scratch/numpy) = {t_sc / t_np:.0f}x")
    return acc_np, acc_sc, t_np, t_sc, max_proj_diff


def main():
    data = load_mnist("data/raw")  # adjust to your data location
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    print(
        f"Loaded MNIST: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}\n"
    )

    # 1) fast sweep to choose r on validation
    val_accs, fit_times = sweep_r_numpy(X_train, y_train, X_val, y_val)
    best_r = R_VALUES[int(np.argmax(val_accs))]
    print(f"\nBest r on validation: {best_r} (val acc {val_accs.max():.4f})")

    # 2) final test accuracy at best_r (numpy engine, full train)
    final_clf = SVDSubspaceClassifier(r=best_r, svd_engine=numpy_svd_engine)
    final_clf.fit(X_train, y_train)
    final_test_acc = final_clf.score(X_test, y_test)
    print(f"FINAL test accuracy at r={best_r}: {final_test_acc:.4f}")

    # 3) scratch-vs-numpy comparison on a subsample at best_r
    X_sub, y_sub = subsample(X_train, y_train, SUBSAMPLE_PER_CLASS)
    acc_np, acc_sc, t_np, t_sc, proj_diff = compare_engines(
        X_sub, y_sub, X_test, y_test, best_r
    )

    # 4) dump for post-processing
    RESULTS_DIR.mkdir(exist_ok=True)
    np.savez(
        RESULTS_DIR / "svd_subspace_sweep.npz",
        r_values=np.array(R_VALUES),
        val_accuracies=val_accs,
        fit_times=fit_times,
        best_r=best_r,
        final_test_acc=final_test_acc,
        compare_acc_numpy=acc_np,
        compare_acc_scratch=acc_sc,
        compare_time_numpy=t_np,
        compare_time_scratch=t_sc,
        compare_max_proj_diff=proj_diff,
    )
    print(f"\nSaved -> {RESULTS_DIR / 'svd_subspace_sweep.npz'}")


if __name__ == "__main__":
    main()
