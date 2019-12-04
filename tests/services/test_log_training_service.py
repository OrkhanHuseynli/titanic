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
            expected = ([[1.0, 1.0],
                         [1.0, 0.9770114942528736],
                         [0.9107142857142857, 0.4827586206896552],
                         [0.8392857142857143, 0.3448275862068966],
                         [0.8035714285714286, 0.22988505747126436],
                         [0.7321428571428571, 0.1724137931034483],
                         [0.6607142857142857, 0.06896551724137931],
                         [0.5, 0.0],
                         [0.375, 0.0],
                         [0.03571428571428571, 0.0],
                         [0.0, 0.0]],
                        [[[56, 87], [0, 0]],
                         [[56, 85], [2, 0]],
                         [[51, 42], [45, 5]],
                         [[47, 30], [57, 9]],
                         [[45, 20], [67, 11]],
                         [[41, 15], [72, 15]],
                         [[37, 6], [81, 19]],
                         [[28, 0], [87, 28]],
                         [[21, 0], [87, 35]],
                         [[2, 0], [87, 54]],
                         [[0, 0], [87, 56]]],
                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
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
