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

    def __init__(self, proccessed_dataset: DataFrame, array_of_X_col: list, y_column_index: int, threshold: int):
        self.processed_dataset = proccessed_dataset
        self.array_of_X_col = array_of_X_col
        self.y_column_index = y_column_index
        self._X_actual = None
        self._y_actual = None
        self.threshold = threshold

    def score(self):
        y_predicted = self.predict(self._X_actual)
        return self._score_(y_predicted, self._y_actual)

    def predict(self, X):
        # return self.__predict_prob(X).round()
        Y = self.__predict_prob(X)
        Y[Y > self.threshold] = 1
        Y[Y <= self.threshold] = 0
        return Y

    def fit(self, alpha, iterations) -> EstimationModel:
        self._theta, self._cost_matrix = self.gradient_descent(alpha, iterations)
        return self

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

    def __predict_prob(self, X):
        return self._logistic_function(np.dot(X, self._theta.T))

    def __gradient_descent__(self, X: Any, y: Any, theta: Any, alpha: int, iterations: int) -> Tuple[Any, np.ndarray]:
        cost = np.zeros(iterations)
        for i in range(iterations):
            h = self._logistic_function(X @ theta.T)
            theta = theta - (alpha / len(X)) * np.sum(X * (h - y), axis=0)
            cost[i] = self.compute_cost(X, y, theta)
            # print("Index ", i)
            # print(theta[0].tolist()[3])
            # print(theta)
        return theta, cost

    def _logistic_function(self, z):
        # sigmoid function
        return 1 / (1 + np.exp(-z))

    def _fit_(self, X: Any, y: Any, theta: Any, alpha: int, iterations: int):
        for i in range(iterations):
            z = np.dot(X, theta)
            h = self._logistic_function(z)
            gradient = np.dot(X.T, (h - y)) / y.shape[0]
            theta -= alpha * gradient

            z = np.dot(X, theta)
            h = self._logistic_function(z)
            loss = self.compute_cost(h, y)
            return theta, loss
