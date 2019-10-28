from unittest import TestCase

import numpy as np
from modules.core.logistic_regression import LogisticRegression
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
