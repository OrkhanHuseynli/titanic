from unittest import TestCase

import pandas as pd

from modules.core.utils import Utils
from modules.services.linear_regression import LinearRegression


class TestLinearRegression(TestCase):
    print("")
    # def test_linear_regression(self):
    #     file_name = "titanic.csv"
    #     test_file_path = '../resources/' + file_name
    #     with open(test_file_path, 'rb') as test_file:
    #         hashed_file_name = Utils.get_hashed_name(file_name) + ".csv"
    #         Utils.write_file(hashed_file_name, test_file.read(), Utils.UPLOADS_FOLDER)
    #         LinearRegression.run_regression(file_name)