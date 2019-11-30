import os
from tempfile import gettempdir
from unittest import TestCase

from modules.core.utils import Utils
from modules.services.log_training_service import LogTrainingService


class TestTrainerService(TestCase):
    def test_train(self):
        file_name = "test.csv"
        with open('../resources/titanic.csv', 'rb') as test_file:
            file = Utils.write_file(file_name, test_file.read(), Utils.UPLOADS_FOLDER)
            alpha = 0.01
            iterations = 1000
            testing_size = 0.8
            tuple_of_X_col, y_column_index = (3, 4, 5, 6, 7), 2
            service = LogTrainingService()
            roc_list = service.train(file_name, testing_size, alpha, iterations, tuple_of_X_col, y_column_index)
            expected = ([[-0.5535714285714286, 0.5],
                         [-0.5178571428571429, 0.4941860465116279],
                         [0.25, 0.3387096774193548],
                         [0.4642857142857143, 0.2777777777777778],
                         [0.6428571428571429, 0.20833333333333334],
                         [0.7321428571428571, 0.1724137931034483],
                         [0.8928571428571429, 0.08108108108108109],
                         [1.0, 0.0],
                         [1.0, 0.0],
                         [1.0, 0.0],
                         [1.0, 0.0]],
                        [[[-31, 87], [87, 87]],
                         [[-29, 85], [87, 85]],
                         [[14, 42], [82, 42]],
                         [[26, 30], [78, 30]],
                         [[36, 20], [76, 20]],
                         [[41, 15], [72, 15]],
                         [[50, 6], [68, 6]],
                         [[56, 0], [59, 0]],
                         [[56, 0], [52, 0]],
                         [[56, 0], [33, 0]],
                         [[56, 0], [31, 0]]])
            self.assertEqual(expected, roc_list)
        Utils.___remove_file___(file.name)

    def test_train_file_doesnt_exist(self):
        with self.assertRaises(Exception) as context:
            file_name = "test.csv"
            alpha = 0.01
            iterations = 1000
            testing_size = 0.8
            tuple_of_X_col, y_column_index = (3, 4, 5, 6, 7), 2
            service = LogTrainingService()
            service.train(file_name, testing_size, alpha, iterations, tuple_of_X_col, y_column_index)
            self.assertTrue((file_name + ' doesnt exist') in context.exception)
