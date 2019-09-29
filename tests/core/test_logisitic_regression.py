from unittest import TestCase

import numpy as np
import pandas as pd

from modules.core.dataset_processor import DatasetProcessor
from modules.core.linear_regression import LinearRegression
from modules.core.logistic_regression import LogisticRegression


class TestLogisticRegression(TestCase):

    def test_score(self):
        log_reg = LogisticRegression([], [], [], False)
        y_ = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
        y = np.array([0, 1, 0, 0, 0, 0, 1, 0, 1, 0])
        actual_score = log_reg._score_(y_, y)
        expected_score = 0.9
        self.assertEqual(expected_score, actual_score)

    def test_predict(self):
        alpha = 0.01
        iterations = 1000
        array_of_X_col, y_column_index = [3, 4, 5, 6, 7], 2
        data_processor = self.__get_data_processor__("titanic.csv", array_of_X_col, y_column_index)
        training_frame, testing_frame = data_processor.split_dataset(0.8)
        log_reg = LogisticRegression(training_frame, array_of_X_col, y_column_index, False)
        log_reg.fit(alpha, iterations)
        print(log_reg.get_coefs())
        actual_score = log_reg.score()
        expected_score = 0.5831873905429071
        self.assertEqual(expected_score, actual_score)

    def __get_data_processor__(self, file_name, array_of_X_col, y_column_index):
        test_file_path = '../resources/' + file_name
        test_dataset = pd.read_csv(test_file_path)
        data_processor = DatasetProcessor(test_dataset)
        # data_processor.normalize()
        return data_processor