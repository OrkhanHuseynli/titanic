from unittest import TestCase

import numpy as np
from modules.core.logistic_regression_util import LogisticRegressionUtil


class TestLogisticRegressionUtil(TestCase):

    def test_confusion_matrix(self):
        y_predicted = np.array([0, 1, 0, 1, 1])
        y_actual = np.array([0, 1, 1, 0, 1])
        actual_confusion_matrix = LogisticRegressionUtil.get_confusion_matrix(y_predicted, y_actual)
        expected_true_positives_count = 2
        expected_true_negatives_count = 1
        expected_false_positives_count = 1
        expected_false_negatives_count = 1
        expected_confusion_matrix = [[expected_true_positives_count, expected_false_positives_count],
                                     [expected_true_negatives_count, expected_false_negatives_count]]
        self.assertEqual(expected_confusion_matrix, actual_confusion_matrix)

    def test_calculate_accuracy(self):
        true_positives = 100
        true_negatives = 50
        false_positives = 10
        false_negatives = 5

        expected = 0.9090909090909091
        actual = LogisticRegressionUtil.calculate_accuracy(true_positives, false_positives,
                                                                     true_negatives, false_negatives)
        self.assertEqual(expected, actual)

    def test_calculate_recall(self):
        true_positives = 100
        false_negatives = 5
        expected = 0.9523809523809523
        actual = LogisticRegressionUtil.calculate_recall(true_positives, false_negatives)
        self.assertEqual(expected, actual)

    def test_calculate_precision(self):
        true_positives = 100
        false_positives = 10

        expected = 0.9090909090909091
        actual = LogisticRegressionUtil.calculate_precision(true_positives, false_positives)
        self.assertEqual(expected, actual)

    def test_calculate_f_measure(self):
        true_positives = 100
        true_negatives = 50
        false_positives = 10
        false_negatives = 5

        expected = 0.9302325581395349
        actual = LogisticRegressionUtil.calculate_f_measure(true_positives, false_positives, false_negatives)
        self.assertEqual(expected, actual)

    def test_calculate_true_positive_rate(self):
        true_positives = 100
        false_negatives = 5

        expected = 0.9523809523809523
        actual = LogisticRegressionUtil.calculate_true_positive_rate(true_positives, false_negatives)
        self.assertEqual(expected, actual)

    def test_calculate_false_positive_rate(self):
        true_negatives = 50
        false_positives = 10
        expected = 0.16666666666666666
        actual = LogisticRegressionUtil.calculate_false_positive_rate(false_positives, true_negatives)
        self.assertEqual(expected, actual)

    def test_calculate_ROC(self):
        y_predicted = np.array([0, 1, 0, 1, 1])
        y_actual = np.array([0, 1, 1, 0, 1])
        expected = [0.6666666666666666, 0.5]
        actual = LogisticRegressionUtil.calculate_ROC(y_predicted, y_actual)
        self.assertEqual(expected, actual)
