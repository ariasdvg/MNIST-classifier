import numpy as np
from linalg.svd import svd as scratch_svd
from .base import Classifier
from data.mnist import stack_by_class


def numpy_svd_engine(A_k, want="U", k=None):
    """Adapter so np.linalg.svd matches the scratch svd's call signature.
    Returns the left singular vectors (U). `want` is accepted for interface
    parity with scratch_svd, but the subspace classifier only ever needs U."""
    U, s, Vh = np.linalg.svd(A_k, full_matrices=False)
    if k is not None:
        U, s = U[:, :k], s[:k]
    return U, s


class SVDSubspaceClassifier(Classifier):
    def __init__(self, r=15, svd_engine=scratch_svd):
        self.r = r
        self.svd_engine = svd_engine
        self.ubases_ = []

    def fit(self, X, y):
        self.ubases_ = []
        for A_k in stack_by_class(X, y).values():
            r_k = min(self.r, *A_k.shape)
            U, s = self.svd_engine(A_k, want="U", k=r_k)
            self.ubases_.append(U)
        return self

    def predict(self, X):
        M = X.shape[0]
        scores = np.empty((M, 10))
        for k in range(10):
            U_k = self.ubases_[k]
            coeffs = X @ U_k
            scores[:, k] = np.sum(coeffs**2, axis=1)
        return np.argmax(scores, axis=1)
