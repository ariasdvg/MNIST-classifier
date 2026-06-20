from abc import ABC, abstractmethod
import numpy as np


class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Train the classifier on labeled data.

        Parameters
        ----------
        X : ndarray, shape (N,d) -> data (Rows: samples; Columns: pixels)
        y : ndarray, shape (N,) -> labels
        """
        return self

    @abstractmethod
    def predict(self, X):
        """Predict labels for samples.

        Parameters
        ----------
        X : ndarray, shape (M, d) -> data (Rows: samples)
        """
        ...

    def score(self, X, y):
        """Scores the accuracy of a classifier"""
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
