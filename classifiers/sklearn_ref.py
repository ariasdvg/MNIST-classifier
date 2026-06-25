import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier

from .base import Classifier


class SklearnRidgeClassifier(Classifier):
    """Golden reference for MultivariateLSClassifier: one-hot ridge REGRESSION
    (sklearn.linear_model.Ridge) + argmax. This mirrors the from-scratch LS
    exactly, with alpha as the regularization strength (alpha->0 = pure LS).

    Note: we use Ridge (the regressor) NOT RidgeClassifier, whose internal
    label encoding misbehaves on this rank-deficient pixel data.
    """

    def __init__(self, alpha=1.0, num_classes=10):
        self.alpha = alpha
        self.num_classes = num_classes
        self._model = Ridge(alpha=alpha)

    def fit(self, X, y):
        Y = np.zeros((y.shape[0], self.num_classes))
        Y[np.arange(y.shape[0]), y] = 1.0
        self._model.fit(X, Y)
        return self

    def predict(self, X):
        return np.argmax(self._model.predict(X), axis=1)


class SklearnKNNClassifier(Classifier):
    """Golden reference for your KNNClassifier."""

    def __init__(self, k=3):
        self._model = KNeighborsClassifier(n_neighbors=k)

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)
