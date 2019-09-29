from typing import Any

from numpy.core.multiarray import ndarray
from pandas import DataFrame

from modules.core.dataset_processor import DatasetProcessor
from modules.core.linear_regression import LinearRegression
from modules.core.api.estimation_model import EstimationModel
from modules.services.api.trainer import Trainer


class TrainerService(Trainer):
    def __init__(self, dataset: DataFrame, array_of_X_col: list, y_column_index: int,
                 training_size: float, alpha: int, iterations: int):
        self.dataset = dataset
        self.training_size = training_size
        self.alpha = alpha
        self.iterations = iterations
        self.array_of_X_col = array_of_X_col
        self.y_column_index = y_column_index
        self.training_frame, self.testing_frame = self.__split_dataset__(dataset, training_size)
        self._coeficients = None
        self._predicted_outcome = None
        self._cost_matrix = None
        self._model: EstimationModel = None
        self._score = None

    def train(self) -> Trainer:
        self._model: EstimationModel = LinearRegression(self.training_frame, self.array_of_X_col, self.y_column_index)
        # NOTE: in an advanced version of the service
        # optimization model can be customizable
        self._model.fit(self.alpha, self.iterations)
        self._score = self._model.score()
        self._coeficients = self._model.get_coefs()
        self._cost_matrix = self._model.get_cost_matrix()
        return self


    def score(self) -> int:
        return self._score

    def get_coefs(self) -> ndarray:
        return self._coeficients

    def get_cost_matrix(self) -> Any:
        return self._cost_matrix

    def predict(self, X: ndarray):
        self._predicted_outcome = self._model.predict(X)
        return self._predicted_outcome

    # @deprecated
    # def predict_on_test_data(self):
    #     X = self.testing_frame
    #     _predicted_on_test_data = self._model.predict(self.testing_frame)

    def __split_dataset__(self, dataset: DataFrame, training_size: int):
        data_processor = DatasetProcessor(dataset)
        # NOTE: in more advanced version of the service,
        # normalization functions will be also configurable
        data_processor.normalize()
        return data_processor.split_dataset(training_size)
