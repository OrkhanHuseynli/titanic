from typing import Any

from numpy.core.multiarray import ndarray
from pandas import DataFrame

from modules.core.dataset_processor import DatasetProcessor
from modules.core.linear_regression import LinearRegression
from modules.services.api.trainer import Trainer


class TrainerService(Trainer):
    def __init__(self, dataset: DataFrame,  array_of_X_col: list, y_column_index: int,
                 training_size: float, alpha: int, iterations: int):
        self.dataset = dataset
        self.training_size = training_size
        self.alpha = alpha
        self.iterations = iterations
        self.array_of_X_col = array_of_X_col
        self.y_column_index = y_column_index
        self.training_frame, self.testing_frame = self.__split_dataset__(dataset, training_size)

    def train(self) -> Trainer:
        lin_reg = LinearRegression(self.training_frame, self.array_of_X_col, self.y_column_index)
        # NOTE: in an advanced version of the service
        # optimization model can be customizable
        gd, cost = lin_reg.gradient_descent(self.alpha, self.iterations)
        pass

    def predict(self, X: ndarray):
        pass

    def __split_dataset__(self, dataset: DataFrame, training_size: int):
        data_processor = DatasetProcessor(dataset)
        # NOTE: in more advanced version of the service,
        # normalization functions will be also configurable
        data_processor.normalize()
        return data_processor.split_dataset(training_size)
