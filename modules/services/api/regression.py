import abc
from typing import Any

from numpy import ndarray

class Regression(metaclass=abc.ABCMeta):
    # intercept_
    # coef_
    @abc.abstractmethod
    def fit(self, x, y) -> Any:
        pass

    # coefficient of determination
    @abc.abstractmethod
    def score(self, x, y) -> int:
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass

    @abc.abstractmethod
    def compute_cost(self, X: ndarray, y: Any, theta: ndarray) -> None:
        pass

    @abc.abstractmethod
    def gradient_descent(self, alpha: int, iterations: int):
        pass
