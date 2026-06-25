import numpy as np
from .base import Classifier


class KNNClassifier(Classifier):
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        # All pairwise squared distances at once: (M_test, N_train)
        dists_sq = (
            np.sum(X**2, axis=1, keepdims=True)
            - 2 * (X @ self.X_train.T)
            + np.sum(self.X_train**2, axis=1)
        )

        # k nearest train indices for each test row, all at once
        nearest = np.argpartition(dists_sq, self.k, axis=1)[:, : self.k]  # (M, k)

        # vote per row
        preds = np.empty(X.shape[0], dtype=self.y_train.dtype)
        for i in range(X.shape[0]):
            labels = self.y_train[nearest[i]]
            preds[i] = np.argmax(np.bincount(labels, minlength=10))
        return preds

