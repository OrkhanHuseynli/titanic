from unittest import TestCase

import numpy as np
import pandas as pd

from modules.core.dataset_processor import DatasetProcessor
from modules.core.logistic_regression import LogisticRegression


class TestLogisticRegression(TestCase):

    def test_score(self):
        log_reg = LogisticRegression([], [], [])
        y_ = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
        y = np.array([0, 1, 0, 0, 0, 0, 1, 0, 1, 0])
        actual = log_reg._score_(y_, y)
        expected = 0.9
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
        expected = 0.7810858143607706
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
        expected = [0.6923076923076923, 0.1572700296735905], [[162, 53], [284, 72]]
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
        expected = 0.7810858143607706
        self.assertEqual(expected, actual)

        # 4 calculate roc points for a list of thresholds on training set
        thresholds_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        roc_list = [None] * len(thresholds_list)
        for i in range(len(thresholds_list)):
            roc_list[i] = log_reg.calculate_roc_point(log_reg.get_X_actual(), log_reg.get_y_actual(),
                                                      thresholds_list[i])

        expected = [([1.0, 1.0], [[234, 337], [0, 0]]),
                    ([1.0, 0.9851632047477745], [[234, 332], [5, 0]]),
                    ([0.8803418803418803, 0.49258160237388726], [[206, 166], [171, 28]]),
                    ([0.8504273504273504, 0.3768545994065282], [[199, 127], [210, 35]]),
                    ([0.7905982905982906, 0.2314540059347181], [[185, 78], [259, 49]]),
                    ([0.6923076923076923, 0.1572700296735905], [[162, 53], [284, 72]]),
                    ([0.6282051282051282, 0.06231454005934718], [[147, 21], [316, 87]]),
                    ([0.49145299145299143, 0.02373887240356083], [[115, 8], [329, 119]]),
                    ([0.29914529914529914, 0.008902077151335312], [[70, 3], [334, 164]]),
                    ([0.017094017094017096, 0.002967359050445104], [[4, 1], [336, 230]]),
                    ([0.0, 0.0], [[0, 0], [337, 234]])]

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
        expected = [([1.0, 1.0], [[56, 87], [0, 0]]),
                    ([1.0, 0.9770114942528736], [[56, 85], [2, 0]]),
                    ([0.9107142857142857, 0.4827586206896552], [[51, 42], [45, 5]]),
                    ([0.8392857142857143, 0.3448275862068966], [[47, 30], [57, 9]]),
                    ([0.8035714285714286, 0.22988505747126436], [[45, 20], [67, 11]]),
                    ([0.7321428571428571, 0.1724137931034483], [[41, 15], [72, 15]]),
                    ([0.6607142857142857, 0.06896551724137931], [[37, 6], [81, 19]]),
                    ([0.5, 0.0], [[28, 0], [87, 28]]),
                    ([0.375, 0.0], [[21, 0], [87, 35]]),
                    ([0.03571428571428571, 0.0], [[2, 0], [87, 54]]),
                    ([0.0, 0.0], [[0, 0], [87, 56]])]

        self.assertEqual(expected, roc_list)

    def __get_data_processor__(self, file_name):
        test_file_path = '../resources/' + file_name
        test_dataset = pd.read_csv(test_file_path)
        data_processor = DatasetProcessor(test_dataset)
        # data_processor.normalize()
        return data_processor
