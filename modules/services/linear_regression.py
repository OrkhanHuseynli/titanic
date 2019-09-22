import array
import os
from typing import Any, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from modules.core.utils import Utils
from modules.services.api.regression import Regression
from modules.services.dataset_processor import DatasetProcessor


class LinearRegression(Regression):
    def __init__(self, proccessed_dataset: DataFrame, array_of_X_col: list, y_column_index: int):
        self.processed_dataset = proccessed_dataset
        self.array_of_X_col = array_of_X_col
        self.y_column_index = y_column_index

    def score(self, y_, y):
        sst = np.sum((y - y.mean()) ** 2)
        ssr = np.sum((y_ - y) ** 2)
        r2 = 1 - (ssr / sst)
        return r2

    def predict(self, x):
        pass

    def fit(self, x, y) -> Regression:
        pass

    def gradient_descent(self, alpha, iterations) -> Tuple[Any, np.ndarray]:
        data_processor = DatasetProcessor(self.processed_dataset)
        data_processor.normalize()
        X, y, theta = data_processor.create_matricies_and_theta(self.array_of_X_col, self.y_column_index)
        return self.__gradient_descent__(X, y, theta, alpha, iterations)

    def __gradient_descent__(self, X: Any, y: Any, theta: Any, alpha: int, iterations: int) -> Tuple[Any, np.ndarray]:
        cost = np.zeros(iterations)
        for i in range(iterations):
            theta = theta - (alpha / len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
            cost[i] = self.compute_cost(X, y, theta)
        return theta, cost

    def compute_cost(self, X, y, theta) -> int:
        for_sum = np.power(((X @ theta.T) - y), 2)
        return np.sum(for_sum) / (2 * len(X))

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
