from unittest import TestCase

import pandas as pd

from modules.services.dataset_processor import DatasetProcessor


class TestDatasetProcessor(TestCase):
    def test_normalize(self):
        file_name = "titanic.csv"
        test_file_path = '../resources/' + file_name
        with open(test_file_path, 'rb') as test_file:
            test_dataset = pd.read_csv(test_file_path)
            processor = DatasetProcessor(test_dataset)
            dataset = processor.get_dataset()
            normalized_dataset = processor.normalize().get_dataset()
            self.assertEqual(dataset.shape, normalized_dataset.shape)

    def test_create_matricies(self):
        file_name = "titanic.csv"
        test_file_path = '../resources/' + file_name
        with open(test_file_path, 'rb') as test_file:
            test_dataset = pd.read_csv(test_file_path)
            processor = DatasetProcessor(test_dataset)
            processor.normalize()
            array_of_x_col = [1, 3]
            # col_names = processor.get_column_names_by_indices(array_of_x_col)
            # print(col_names)
            processor.create_matricies_and_theta(array_of_x_col, 5)

    def test__split_dataset_(self):
        file_name = "training_data.csv"
        test_file_path = '../resources/' + file_name
        with open(test_file_path, 'rb') as test_file:
            test_dataset = pd.read_csv(test_file_path)
            processor = DatasetProcessor(test_dataset)
            tr_df, ts_df = processor.__split_dataset__(processor.dataset, 0.8)
            print(tr_df)
            print(ts_df)