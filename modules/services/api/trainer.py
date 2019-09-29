import abc
from typing import Any

from numpy.core.multiarray import ndarray

class Trainer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(self) -> Any:
        pass

    @abc.abstractmethod
    def score(self) -> int:
        pass

    @abc.abstractmethod
    def predict(self, X: ndarray):
        pass

    @abc.abstractmethod
    def get_coefs(self) -> ndarray:
        pass

    @abc.abstractmethod
    def get_cost_matrix(self) -> Any:
        pass