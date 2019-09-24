from unittest import TestCase

import numpy as np
import pandas as pd

from modules.core.dataset_processor import DatasetProcessor
from modules.core.linear_regression import LinearRegression


class TestLinearRegression(TestCase):
    def test_compute_cost(self):
        alpha = 0.01
        array_of_X_col, y_column_index = [0, 1], 2
        data_processor, lin_reg = self.__get_data_processor_and_lin_reg__("training_data.csv", array_of_X_col, y_column_index)
        X, y, theta = data_processor.create_matricies_and_theta(array_of_X_col, y_column_index)
        theta = theta - (alpha / len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost = lin_reg.compute_cost(X, y, theta)
        expected_cost = 0.48054910410767177
        self.assertEqual(expected_cost, cost)

    def test_gradient_descent(self):
        alpha = 0.01
        iterations = 1000
        array_of_X_col, y_column_index = [0, 1], 2
        data_processor, lin_reg = self.__get_data_processor_and_lin_reg__("training_data.csv", array_of_X_col, y_column_index)
        gd, cost = lin_reg.gradient_descent(alpha, iterations)
        print("gradient decent: ")
        print(gd)
        X_, y_, theta_ = data_processor.create_matricies_and_theta([0, 1], 2)
        final_cost = lin_reg.compute_cost(X_, y_, gd)
        print(final_cost)
        expected_final_cost = 0.1307033696077189
        self.assertEqual(expected_final_cost, final_cost)

    def test_gradient_descent2(self):
        alpha = 0.01
        iterations = 1000
        array_of_X_col, y_column_index = [0, 1, 3], 2
        data_processor, lin_reg = self.__get_data_processor_and_lin_reg__("titanic.csv", array_of_X_col, y_column_index)
        gd, cost = lin_reg.gradient_descent(alpha, iterations)
        print("gradient decent: ")
        print(gd)
        X_, y_, theta_ = data_processor.create_matricies_and_theta([0, 1, 3], 2)
        final_cost = lin_reg.compute_cost(X_, y_, gd)
        print(final_cost)

    def test_predict(self):
        alpha = 0.01
        iterations = 1000
        array_of_X_col, y_column_index = [0, 1], 2
        data_processor, lin_reg = self.__get_data_processor_and_lin_reg__("training_data.csv", array_of_X_col, y_column_index)
        lin_reg.fit(alpha, iterations)
        data_processor_, lin_reg_ = self.__get_data_processor_and_lin_reg__("testing_data.csv", [0, 1], 2)
        X_, y_, theta_ = data_processor_.create_matricies_and_theta([0, 1], 2)
        y_pred = lin_reg.predict(X_)
        r2 = lin_reg.score(y_pred, y_)
        print(r2)

    def __get_data_processor_and_lin_reg__(self, file_name, array_of_X_col, y_column_index):
        test_file_path = '../resources/' + file_name
        test_dataset = pd.read_csv(test_file_path)
        data_processor = DatasetProcessor(test_dataset)
        data_processor.normalize()
        lin_reg = LinearRegression(data_processor.get_dataset(), array_of_X_col, y_column_index)
        return data_processor, lin_reg


# y_pred = model.intercept_ + model.coef_ * x