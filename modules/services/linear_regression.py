import os

import numpy as np
import pandas as pd

from modules.core.utils import Utils


class LinearRegression:
    @staticmethod
    def run_regression(file_name):
        dataset = LinearRegression.__read_dataset__(file_name)
        print(dataset.head())

    @staticmethod
    def __read_dataset__(file_name):
        file_path = Utils.get_stored_file_path(file_name, Utils.UPLOADS_FOLDER)
        return pd.read_csv(file_path)

