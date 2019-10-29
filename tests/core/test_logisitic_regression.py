from unittest import TestCase

import numpy as np
import pandas as pd

from modules.core.dataset_processor import DatasetProcessor
from modules.core.logistic_regression import LogisticRegression


class TestLogisticRegression(TestCase):

    def test_score(self):
        log_reg = LogisticRegression([], [], [], 0.5)
        y_ = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
        y = np.array([0, 1, 0, 0, 0, 0, 1, 0, 1, 0])
        actual = log_reg._score_(y_, y)
        expected = 0.8181818181818182
        self.assertEqual(expected, actual)

    def test_predict_with_binary(self):
        alpha = 0.01
        iterations = 1000
        threshold = 0.5
        array_of_X_col, y_column_index = [3, 4, 5, 6, 7], 2
        data_processor = self.__get_data_processor__("titanic.csv")
        training_frame, testing_frame = data_processor.split_dataset(0.8)
        log_reg = LogisticRegression(training_frame, array_of_X_col, y_column_index)
        log_reg.fit(alpha, iterations)
        actual = log_reg.score()
        expected = 0.8079710144927537
        self.assertEqual(expected, actual)
        data_processor_testing_frame = DatasetProcessor(testing_frame)
        X_test, y_test, _ = data_processor_testing_frame.create_matricies_and_theta_for_binary_output([3, 4, 5, 6, 7],
                                                                                                      2)
        y_test_predicted = log_reg.predict(X_test, threshold)
        test_score = log_reg._score_(y_test_predicted, y_test)
        test_score_expected = 0.7902097902097902
        self.assertEqual(test_score_expected, test_score)

    def test_calculate_roc_point(self):
        alpha = 0.01
        iterations = 1000
        threshold = 0.5
        array_of_X_col, y_column_index = [3, 4, 5, 6, 7], 2
        data_processor = self.__get_data_processor__("titanic.csv")
        training_frame, testing_frame = data_processor.split_dataset(0.8)
        log_reg = LogisticRegression(training_frame, array_of_X_col, y_column_index)
        log_reg.fit(alpha, iterations)
        actual = log_reg.calculate_roc_point(log_reg.get_X_actual(), log_reg.get_y_actual(), threshold)
        expected = [0.7735042735042735, 0.16666666666666666]
        self.assertEqual(expected, actual)

    def test_full_cycle(self):
        # 1 split data into training and testing data
        split_ratio = 0.8
        data_processor = self.__get_data_processor__("titanic.csv")
        training_frame, testing_frame = data_processor.split_dataset(split_ratio)
        # 2 define input params and outcome param (select necessary columns)
        alpha = 0.01
        iterations = 1000
        array_of_X_col, y_column_index = [3, 4, 5, 6, 7], 2

        # 3 train the model with the training data
        log_reg = LogisticRegression(training_frame, array_of_X_col, y_column_index)
        log_reg.fit(alpha, iterations)
        actual = log_reg.score()
        expected = 0.8079710144927537
        self.assertEqual(expected, actual)

        # 4 calculate roc points for a list of thresholds on training set
        thresholds_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        roc_list = [None] * len(thresholds_list)
        for i in range(len(thresholds_list)):
            roc_list[i] = log_reg.calculate_roc_point(log_reg.get_X_actual(), log_reg.get_y_actual(),
                                                      thresholds_list[i])
        expected = [[-0.44017094017094016, 0.5],
                    [-0.4188034188034188, 0.4962630792227205],
                    [0.2905982905982906, 0.3494736842105263],
                    [0.45726495726495725, 0.29603729603729606],
                    [0.6666666666666666, 0.21311475409836064],
                    [0.7735042735042735, 0.16666666666666666],
                    [0.9102564102564102, 0.07749077490774908],
                    [0.9658119658119658, 0.035398230088495575],
                    [0.9871794871794872, 0.017045454545454544],
                    [0.9957264957264957, 0.009259259259259259],
                    [1.0, 0.0]]
        self.assertEqual(expected, roc_list)

        # 5 test the model with the testing data
        threshold = 0.5
        data_processor_testing_frame = DatasetProcessor(testing_frame)
        X_test_actual, y_test_actual, _ = data_processor_testing_frame.create_matricies_and_theta_for_binary_output(
            [3, 4, 5, 6, 7], 2)
        y_test_predicted = log_reg.predict(X_test_actual, threshold)
        actual = log_reg._score_(y_test_predicted, y_test_actual)
        expected = 0.7902097902097902
        self.assertEqual(expected, actual)

        # 6 calculate roc points for a list of thresholds on test set
        roc_list = [None] * len(thresholds_list)
        for i in range(len(thresholds_list)):
            roc_list[i] = log_reg.calculate_roc_point(X_test_actual, y_test_actual, thresholds_list[i])
        expected = [[-0.5535714285714286, 0.5],
                    [-0.5178571428571429, 0.4941860465116279],
                    [0.25, 0.3387096774193548],
                    [0.4642857142857143, 0.2777777777777778],
                    [0.6428571428571429, 0.20833333333333334],
                    [0.7321428571428571, 0.1724137931034483],
                    [0.8928571428571429, 0.08108108108108109],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0]]
        self.assertEqual(expected, roc_list)

    def __get_data_processor__(self, file_name):
        test_file_path = '../resources/' + file_name
        test_dataset = pd.read_csv(test_file_path)
        data_processor = DatasetProcessor(test_dataset)
        # data_processor.normalize()
        return data_processor
