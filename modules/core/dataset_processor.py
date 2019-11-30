from typing import Any, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame


class DatasetProcessor:
    def __init__(self, dataset: DataFrame):
        self.original_dataset = dataset
        self.dataset = dataset

    def return_columns(self):
        return self.dataset.columns

    def get_dataset(self) -> DataFrame:
        return self.dataset

    def normalize(self):
        self.dataset = (self.dataset - self.dataset.mean()) / self.dataset.std()
        return self


    def _normalize_dataset(self, dataset) -> DataFrame:
        return (dataset - dataset.mean()) / dataset.std()

    def create_matricies_and_theta_for_binary_output(self, array_of_X_col: list, y_column_index: int) -> Tuple[ndarray, Any, ndarray]:
        array_of_X_col.append(y_column_index)
        full_array_of_variables = array_of_X_col.copy()
        self.transform_indicies_to_colnames(full_array_of_variables)
        # create a new dataset, where the last column is the dependent variable
        input_dataset = self.dataset[full_array_of_variables[:-1]]
        normalized_input_dataset = self._normalize_dataset(input_dataset)
        output_dataset = self.dataset[full_array_of_variables[-1:]]
        new_dataset_with_normalized_inputs = pd.concat([normalized_input_dataset, output_dataset], axis=1, sort=False)
        last_colindex = len(full_array_of_variables)
        # set matrixes
        X = new_dataset_with_normalized_inputs.iloc[:, 0:last_colindex - 1]
        ones = np.ones([X.shape[0], 1])
        X = np.concatenate((ones, X), axis=1)
        y = new_dataset_with_normalized_inputs.iloc[:, last_colindex - 1:last_colindex].values
        theta = np.zeros([1, last_colindex])
        return X, y, theta

    def create_matricies_and_theta(self, array_of_X_col: list, y_column_index: int) -> Tuple[ndarray, Any, ndarray]:
        array_of_X_col.append(y_column_index)
        full_array_of_variables = array_of_X_col.copy()
        self.transform_indicies_to_colnames(full_array_of_variables)
        # create a new dataset, where the last column is the dependent variable
        new_dataset = self.dataset[full_array_of_variables]
        last_colindex = len(full_array_of_variables)
        # set matrixes
        X = new_dataset.iloc[:, 0:last_colindex - 1]
        ones = np.ones([X.shape[0], 1])
        X = np.concatenate((ones, X), axis=1)
        y = new_dataset.iloc[:, last_colindex - 1:last_colindex].values
        theta = np.zeros([1, last_colindex])
        return X, y, theta

    def transform_indicies_to_colnames(self, indices: list) -> list:
        for i in range(len(indices)):
            indices[i] = self.dataset.columns[indices[i]]
        return indices

    #   Tuple[ndarray, Any, ndarray, Any]
    def split_dataset(self, training_size: float):
        return self.__split_dataset__(self.dataset, training_size)

    def __split_dataset__(self, data_frame: DataFrame, training_size: float) -> Tuple[DataFrame, DataFrame]:
        split_index = round(training_size * len(data_frame))
        training_frame, testing_frame = np.split(data_frame, [split_index], 0)
        return training_frame, testing_frame
