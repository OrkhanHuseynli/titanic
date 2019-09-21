from unittest import TestCase

import pandas as pd

from modules.core.utils import Utils
from modules.services.linear_regression import LinearRegression


class TestLinearRegression(TestCase):
    def test__read_dataset__(self):
        file_name = "titanic.csv"
        test_file_path = '../resources/'+file_name
        with open(test_file_path, 'rb') as test_file:
            test_dataset = pd.read_csv(test_file_path)
            hashed_file_name = Utils.get_hashed_name(file_name) + ".csv"
            Utils.write_file(hashed_file_name, test_file.read(), Utils.UPLOADS_FOLDER)
            dataset = LinearRegression.__read_dataset__(file_name)
            self.assertEqual(test_dataset.shape, dataset.shape)
            Utils.remove_file(hashed_file_name, Utils.UPLOADS_FOLDER)