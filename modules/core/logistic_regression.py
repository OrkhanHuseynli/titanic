from typing import Any, Tuple

import numpy as np
from pandas import DataFrame

from modules.core.api.estimation_model import EstimationModel
from modules.core.dataset_processor import DatasetProcessor
from modules.core.logistic_regression_util import LogisticRegressionUtil


class LogisticRegression(EstimationModel):
    # theta - intercept and coeficient values
    _theta = None
    _cost_matrix = None

    def __init__(self, processed_dataset: DataFrame, array_of_X_col: list, y_column_index: int):
        self.processed_dataset = processed_dataset
        self.array_of_X_col = array_of_X_col
        self.y_column_index = y_column_index
        self._X_actual = None
        self._y_actual = None

    def score(self):
        y_predicted = self.predict(self._X_actual, 0.5)
        return self._score_(y_predicted, self._y_actual)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        # return self.__predict_prob(X).round()
        Y = self.__predict_prob__(X)
        Y[Y >= threshold] = 1
        Y[Y < threshold] = 0
        return Y

    def calculate_roc_point(self, X_actual: np.ndarray, y_actual: np.ndarray, threshold: int) -> (list, list):
        y_predicted = self.predict(X_actual, threshold)
        return LogisticRegressionUtil.calculate_ROC(y_predicted, y_actual)

    def fit(self, alpha, iterations) -> EstimationModel:
        self._theta, self._cost_matrix = self.gradient_descent(alpha, iterations)
        return self

    def get_X_actual(self):
        return self._X_actual

    def get_y_actual(self):
        return self._y_actual

    def get_coefs(self):
        return self._theta

    def get_cost_matrix(self):
        return self._cost_matrix

    def gradient_descent(self, alpha: int, iterations: int) -> Tuple[Any, np.ndarray]:
        data_processor = DatasetProcessor(self.processed_dataset)
        self._X_actual, self._y_actual, theta = data_processor.create_matricies_and_theta_for_binary_output(
            self.array_of_X_col, self.y_column_index)
        return self.__gradient_descent__(self._X_actual, self._y_actual, theta, alpha, iterations)

    def compute_cost(self, X, y, theta) -> int:
        for_sum = np.power(((X @ theta.T) - y), 2)
        return np.sum(for_sum) / (2 * len(X))


    def _score_(self, y_predicted, y_actual):
        confusion_matrix = LogisticRegressionUtil.get_confusion_matrix(y_predicted, y_actual)
        accuracy = LogisticRegressionUtil.calculate_accuracy(confusion_matrix[0][0], confusion_matrix[0][1],
                                                             confusion_matrix[1][0], confusion_matrix[1][1])
        return accuracy

    def __predict_prob__(self, X):
        return self._logistic_function_(np.dot(X, self._theta.T))

    def __gradient_descent__(self, X: Any, y: Any, theta: Any, alpha: int, iterations: int) -> Tuple[Any, np.ndarray]:
        cost = np.zeros(iterations)
        for i in range(iterations):
            h = self._logistic_function_(X @ theta.T)
            theta = theta - (alpha / len(X)) * np.sum(X * (h - y), axis=0)
            cost[i] = self.compute_cost(X, y, theta)
            # print("Index ", i)
            # print(theta[0].tolist()[3])
            # print(theta)
        return theta, cost

    def _logistic_function_(self, z):
        # sigmoid function
        return 1 / (1 + np.exp(-z))
