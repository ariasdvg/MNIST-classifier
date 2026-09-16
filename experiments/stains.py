import numpy as np
import warnings
import sys
from pathlib import Path

# Put the repo root on sys.path so `python experiments/<script>.py` works
# as well as `python -m experiments.<script>`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.mnist import load_mnist
from classifiers.svd_subspace import SVDSubspaceClassifier, numpy_svd_engine
from linalg.svd import svd as scratch_svd

data = load_mnist("data/raw")
Xtr, ytr = data["X_train"], data["y_train"]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # silence the fallback noise
    clf_sc = SVDSubspaceClassifier(r=30, svd_engine=scratch_svd).fit(Xtr, ytr)
clf_np = SVDSubspaceClassifier(r=30, svd_engine=numpy_svd_engine).fit(Xtr, ytr)

worst = 0.0
for Uk_sc, Uk_np in zip(clf_sc.ubases_, clf_np.ubases_):
    P_sc, P_np = Uk_sc @ Uk_sc.T, Uk_np @ Uk_np.T
    worst = max(worst, np.max(np.abs(P_sc - P_np)))
print("worst top-30 projector diff (full data):", worst)
