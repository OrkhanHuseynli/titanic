from unittest import TestCase

import numpy as np
import pandas as pd

from modules.core.utils import Utils
from modules.services.dataset_processor import DatasetProcessor
from modules.services.linear_regression import LinearRegression


class TestLinearRegression(TestCase):
    def test_compute_cost(self):
        alpha = 0.01
        array_of_X_col, y_column_index = [0, 1], 2
        data_processor, lin_reg = self.__get_data_processor_and_lin_reg__(array_of_X_col, y_column_index)
        X, y, theta = data_processor.create_matricies_and_theta(array_of_X_col, y_column_index)
        theta = theta - (alpha / len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost = lin_reg.compute_cost(X, y, theta)
        expected_cost = 0.48054910410767177
        self.assertEqual(expected_cost, cost)

    def test_gradient_descent(self):
        alpha = 0.01
        iterations = 1000
        array_of_X_col, y_column_index = [0, 1], 2
        data_processor, lin_reg = self.__get_data_processor_and_lin_reg__(array_of_X_col, y_column_index)
        gd, cost = lin_reg.gradient_descent(alpha, iterations)
        print(gd)
        X_, y_, theta_ = data_processor.create_matricies_and_theta([0, 1], 2)
        final_cost = lin_reg.compute_cost(X_, y_, gd)
        print(final_cost)
        expected_final_cost = 0.13070336960771892
        self.assertEqual(expected_final_cost, final_cost)

    def __get_data_processor_and_lin_reg__(self, array_of_X_col, y_column_index):
        file_name = "test_data.csv"
        test_file_path = '../resources/' + file_name
        test_dataset = pd.read_csv(test_file_path)
        data_processor = DatasetProcessor(test_dataset)
        data_processor.normalize()
        lin_reg = LinearRegression(data_processor.get_dataset(), array_of_X_col, y_column_index)
        return data_processor, lin_reg
