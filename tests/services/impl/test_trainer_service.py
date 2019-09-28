from unittest import TestCase

import numpy as np
import pandas as pd

from modules.core.dataset_processor import DatasetProcessor
from modules.core.linear_regression import LinearRegression
from modules.services.api.trainer import Trainer
from modules.services.impl.trainer_service import TrainerService


class TestTrainerService(TestCase):
    def test_train(self):
        alpha = 0.01
        iterations = 1000
        array_of_X_col, y_column_index = [0, 1], 2
        service: Trainer = TrainerService(self.__get_dataset__("test_data.csv"), array_of_X_col, y_column_index, 0.8, alpha, iterations)
        service.train()
        print("coeficients : ", service.get_coefs()[0])
        self.assertEqual([0.039826403664998934, 0.9587822599661973, -0.00246734131974181], service.get_coefs()[0].tolist())

    def test_predict(self):
        alpha = 0.01
        iterations = 1000
        array_of_X_col, y_column_index = [0, 1], 2
        service = TrainerService(self.__get_dataset__("test_data.csv"), array_of_X_col, y_column_index, 0.8, alpha, iterations)
        service.train()
        # data_processor, lin_reg = self.__get_data_processor_and_lin_reg__("training_data.csv", array_of_X_col,
        # y_column_index) lin_reg.fit(alpha, iterations) data_processor_, lin_reg_ =
        # self.__get_data_processor_and_lin_reg__("testing_data.csv", [0, 1], 2) X_, y_, theta_ =
        # data_processor_.create_matricies_and_theta([0, 1], 2) y_pred = lin_reg.predict(X_) r2 = lin_reg.score(
        # y_pred, y_) print(r2)

    def __get_dataset__(self, file_name):
        test_file_path = '../../resources/' + file_name
        return pd.read_csv(test_file_path)

