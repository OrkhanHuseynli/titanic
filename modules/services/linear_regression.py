import os

import numpy as np
import pandas as pd

from modules.core.utils import Utils
from modules.services.dataset_processor import DatasetProcessor


class LinearRegression:
    def __init__(self, proccessed_dataset, array_of_X_col, y_column_index):
        self.processed_dataset = proccessed_dataset
        self.array_of_X_col = array_of_X_col
        self.y_column_index = y_column_index

    #Gradient descent
    def gradient_descent(self, alpha, iterations):
        data_processor = DatasetProcessor(self.processed_dataset)
        data_processor.normalize()
        X, y, theta = data_processor.create_matricies_and_theta(self.array_of_X_col, self.y_column_index)
        return self.__gradient_descent__(X, y, theta, alpha, iterations)


    #Gradient descent
    def __gradient_descent__(self, X, y, theta, alpha, iterations):
        cost = np.zeros(iterations)
        for i in range(iterations):
            theta = theta - (alpha / len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
            cost[i] = self.compute_cost(X, y, theta)
        return theta, cost

    #Cost function
    def compute_cost(self, X, y, theta):
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


