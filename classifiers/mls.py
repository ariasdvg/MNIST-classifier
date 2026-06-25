import numpy as np
from .base import Classifier


def _one_hot(y, num_classes=10):
    """Encode the integer labels as an one-hot target matrix"""
    Y = np.zeros((y.shape[0], num_classes))
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


class MultivariateLSClassifier(Classifier):
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.B_ = None  # learn matrix

    def fit(self, X, y):
        Y = _one_hot(y, self.num_classes)
        X_pinv = np.linalg.pinv(X)  # pseudoinverse from data matrix
        self.B_ = X_pinv @ Y
        return self

    def predict(self, X):
        return np.argmax(X @ self.B_, axis=1)  # calculate the highest score per row
