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

    # def test_predict(self):
    #     alpha = 0.01
    #     iterations = 1000
    #     array_of_X_col, y_column_index = [0, 1], 2
    #     data_processor, lin_reg = self.__get_data_processor_and_lin_reg__("training_data.csv", array_of_X_col, y_column_index)
    #     lin_reg.fit(alpha, iterations)
    #     data_processor_, lin_reg_ = self.__get_data_processor_and_lin_reg__("testing_data.csv", [0, 1], 2)
    #     X_, y_, theta_ = data_processor_.create_matricies_and_theta([0, 1], 2)
    #     y_pred = lin_reg.predict(X_)
    #     r2 = lin_reg._score_(y_pred, y_)
    #     print(r2)

    def test_score(self):
        alpha = 0.01
        iterations = 1000
        array_of_X_col, y_column_index = [0, 1], 2
        data_processor, lin_reg = self.__get_data_processor_and_lin_reg__("training_data.csv", array_of_X_col, y_column_index)
        lin_reg.fit(alpha, iterations)
        expected_score = 0.7329105055842267
        actual_score = lin_reg.score()
        self.assertEqual(expected_score, actual_score)

    def test_predict(self):
        alpha = 0.01
        iterations = 1000
        array_of_X_col, y_column_index = [3, 4, 5], 2
        data_processor = self.__get_data_processor__("titanic.csv", array_of_X_col, y_column_index)
        training_frame, testing_frame = data_processor.split_dataset(0.8)
        lin_reg = LinearRegression(training_frame, array_of_X_col, y_column_index)
        lin_reg.fit(alpha, iterations)
        actual_score = lin_reg.score()
        print(actual_score)
        # expected_score = 0.7615686264521379
        # self.assertEqual(expected_score, actual_score)
        data_processor_2 = DatasetProcessor(testing_frame)
        X_test, y_test,  _ = data_processor_2.create_matricies_and_theta([0, 1], 2)
        y_predicted = lin_reg.predict(X_test)
        test_score = lin_reg._score_(y_predicted, y_test)
        print(test_score)


    def __get_data_processor_and_lin_reg__(self, file_name, array_of_X_col, y_column_index):
        data_processor = self.__get_data_processor__(file_name, array_of_X_col, y_column_index)
        lin_reg = LinearRegression(data_processor.get_dataset(), array_of_X_col, y_column_index)
        return data_processor, lin_reg

    def __get_data_processor__(self, file_name, array_of_X_col, y_column_index):
        test_file_path = '../resources/' + file_name
        test_dataset = pd.read_csv(test_file_path)
        data_processor = DatasetProcessor(test_dataset)
        data_processor.normalize()
        return data_processor

# y_pred = model.intercept_ + model.coef_ * x