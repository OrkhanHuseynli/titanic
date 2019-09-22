import array
import os
from typing import Any, Tuple

from numpy import ndarray
import numpy as np
import pandas as pd
from pandas import DataFrame

from modules.core.utils import Utils


class DatasetProcessor:
    def __init__(self, dataset: DataFrame):
        self.dataset = dataset

    def return_columns(self):
        return self.dataset.columns

    def get_dataset(self) -> DataFrame:
        return self.dataset

    def normalize(self):
        self.dataset = (self.dataset - self.dataset.mean()) / self.dataset.std()
        return self

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
        # set hyper parameters
        # alpha = 0.01
        # iters = 1000
        return X, y, theta

    def transform_indicies_to_colnames(self, indices: list) -> list:
        for i in range(len(indices)):
            indices[i] = self.dataset.columns[indices[i]]
        return indices
