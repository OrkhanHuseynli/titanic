from typing import Any, Tuple

import numpy as np
from pandas import DataFrame

from modules.core.api.estimation_model import EstimationModel
from modules.core.dataset_processor import DatasetProcessor


class LogisticRegression(EstimationModel):
    # theta - intercept and coeficient values
    _theta = None
    _cost_matrix = None

    def __init__(self, proccessed_dataset: DataFrame, array_of_X_col: list, y_column_index: int, verbose):
        self.processed_dataset = proccessed_dataset
        self.array_of_X_col = array_of_X_col
        self.y_column_index = y_column_index
        self._X_actual = None
        self._y_actual = None
        self.verbose = verbose

    def score(self):
        y_predicted = self.predict(self._X_actual)
        return self._score_(y_predicted, self._y_actual)

    def _score_(self, y_, y):
        # calculating the ratio of correct predictions
        ssr = np.sum((y_ - y) ** 2)
        accuracy = (y.size - ssr)/y.size
        return accuracy

    def predict(self, X):
        return self.__predict_prob(X).round()

    def __predict_prob(self, X):
        return self._logistic_function(np.dot(X, self._theta.T))

    # def predict_on_test_dataset(self):
    #     data_processor = DatasetProcessor(self.processed_dataset)
    #     # data_processor.normalize()
    #     X, y, theta = data_processor.create_matricies_and_theta(self.array_of_X_col, self.y_column_index)


    def fit(self, alpha, iterations) -> EstimationModel:
        # data_processor = DatasetProcessor(self.processed_dataset)
        # self._X_actual, self._y_actual, self._theta = data_processor.create_matricies_and_theta_for_binary_output(self.array_of_X_col, self.y_column_index)
        # self._theta, self._cost_matrix =self._fit_(self._X_actual, self._y_actual, self._theta, alpha, iterations)
        self._theta, self._cost_matrix = self.gradient_descent(alpha, iterations)
        return self

    def get_coefs(self):
        return self._theta

    def get_cost_matrix(self):
        return self._cost_matrix

    def gradient_descent(self, alpha: int, iterations: int) -> Tuple[Any, np.ndarray]:
        data_processor = DatasetProcessor(self.processed_dataset)
        self._X_actual, self._y_actual, theta = data_processor.create_matricies_and_theta_for_binary_output(self.array_of_X_col, self.y_column_index)
        return self.__gradient_descent__(self._X_actual, self._y_actual, theta, alpha, iterations)

    def compute_cost(self, X, y, theta) -> int:
        for_sum = np.power(((X @ theta.T) - y), 2)
        return np.sum(for_sum) / (2 * len(X))

    def __gradient_descent__(self, X: Any, y: Any, theta: Any, alpha: int, iterations: int) -> Tuple[Any, np.ndarray]:
        cost = np.zeros(iterations)
        for i in range(iterations):
            # if i==303:
            #     print("nr")
            h = self._logistic_function(X @ theta.T)
            theta = theta - (alpha / len(X)) * np.sum(X * (h - y), axis=0)
            cost[i] = self.compute_cost(X, y, theta)
            # print("Index ", i)
            # print(theta[0].tolist()[3])
            # print(theta)
        return theta, cost

    def _logistic_function(self, z):
        #sigmoid function
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
    # def compute_cost(self, h, y):
    #     return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    # def transform_indicies_to_colnames(self, indices):
    #     for i in range(len(indices)):
    #         indices[i] = self.dataset.columns[indices[i]]
    #     return indices

    # @staticmethod
    # def run_regression(file_name):
    #     dataset = LinearRegression.__read_dataset__(file_name)
    #     col = dataset.columns
    #     print(col)
    #     print(col[2])
    #     # initializing our inputs and outputs
    #     X = dataset[col[2]].values
    #     print(X)
    #     # mean of our inputs and outputs
    #     x_mean = np.mean(X)
    #     # y_mean = np.mean(Y)
    #     # Y = dataset['Brain Weight(grams)'].values
    #