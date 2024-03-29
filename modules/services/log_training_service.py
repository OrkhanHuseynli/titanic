import os
import string
from tempfile import gettempdir
from typing import Any

import pandas as pd
from numpy.core.multiarray import ndarray
from pandas import DataFrame

from modules.core.dataset_processor import DatasetProcessor
from modules.core.linear_regression import LinearRegression
from modules.core.api.estimation_model import EstimationModel
from modules.core.logistic_regression import LogisticRegression
from modules.core.utils import Utils
from modules.services.api.trainer import Trainer


class LogTrainingService(Trainer):
    def __init__(self):
        self._coeficients = None
        self._predicted_outcome = None
        self._model: EstimationModel = None

    def get_coefs(self) -> ndarray:
        return self._coeficients

    def predict(self, X: ndarray):
        return None

    def train(self, filename: string, testing_size: int, alpha: int, iterations: int, array_of_X_col: tuple,
              y_column_index: int) -> (list, list, list):
        # 1 split data into training and testing data
        try:
            dataset = self.__get_raw_data__(filename)
        except OSError:
            raise OSError(filename, ' doesnt exist')

        data_processor = DatasetProcessor(dataset)
        training_frame, testing_frame = data_processor.split_dataset(testing_size)
        # 2 Build logistic regression and hence the model
        log_reg = LogisticRegression(training_frame, list(array_of_X_col), y_column_index)
        log_reg.fit(alpha, iterations)
        # 3 start working on the testing data
        data_processor_testing_frame = DatasetProcessor(testing_frame)
        X_test_actual, y_test_actual, _ = data_processor_testing_frame.create_matricies_and_theta_for_binary_output(
            list(array_of_X_col), y_column_index)
        # 4 calculate roc points for a list of thresholds on test set
        thresholds_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        roc_list = [None] * len(thresholds_list)
        conf_matrix_list = [None] * len(thresholds_list)
        print("calculating roc")
        for i in range(len(thresholds_list)):
            roc_list[i], conf_matrix_list[i] = log_reg.calculate_roc_point(X_test_actual, y_test_actual,
                                                                           thresholds_list[i])
        return roc_list, conf_matrix_list, thresholds_list

    def __get_raw_data__(self, file_name) -> DataFrame:
        test_file_path = os.path.join(gettempdir(), Utils.UPLOADS_FOLDER, file_name)
        return pd.read_csv(test_file_path)
