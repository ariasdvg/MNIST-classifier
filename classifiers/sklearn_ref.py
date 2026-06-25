import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from .base import Classifier


class SklearnRidgeClassifier(Classifier):
    """Golden reference for your LeastSquaresClassifier (one-hot ridge regression)."""

    def __init__(self, alpha=1.0):
        self._model = RidgeClassifier(alpha=alpha)

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


class SklearnKNNClassifier(Classifier):
    """Golden reference for your KNNClassifier."""

    def __init__(self, k=3):
        self._model = KNeighborsClassifier(n_neighbors=k)

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)
