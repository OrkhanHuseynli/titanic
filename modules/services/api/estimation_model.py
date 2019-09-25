import abc
from typing import Any, Tuple

from numpy import ndarray


class EstimationModel(metaclass=abc.ABCMeta):
    # intercept_
    # coef_
    @abc.abstractmethod
    def fit(self, x, y) -> Any:
        pass

    @abc.abstractmethod
    def get_coefs(self) -> ndarray:
        pass

    @abc.abstractmethod
    def get_cost_matrix(self) -> Any:
        pass

    @abc.abstractmethod
    def score(self, x, y) -> int:
        pass

    @abc.abstractmethod
    def predict(self, x: ndarray) -> ndarray:
        pass

    @abc.abstractmethod
    def compute_cost(self, X: ndarray, y: Any, theta: ndarray) -> None:
        pass
