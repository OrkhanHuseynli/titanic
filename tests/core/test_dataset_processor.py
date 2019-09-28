from unittest import TestCase

import pandas as pd

from modules.core.dataset_processor import DatasetProcessor


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
            array_of_x_col, y_col = [3, 4, 5], 2
            X, y, theta = processor.create_matricies_and_theta_for_binary_output(array_of_x_col, y_col)
            expected_output_values = [[-0.8264407624318818], [1.2083133905900616], [1.2083133905900616]]
            actual_output_values = y.tolist()[:3]
            self.assertEqual(expected_output_values, actual_output_values)

    def test_create_matricies_for_binary_output(self):
        file_name = "titanic.csv"
        test_file_path = '../resources/' + file_name
        with open(test_file_path, 'rb') as test_file:
            test_dataset = pd.read_csv(test_file_path)
            processor = DatasetProcessor(test_dataset)
            array_of_x_col, y_col = [3, 4, 5], 2
            X, y, theta = processor.create_matricies_and_theta_for_binary_output(array_of_x_col, y_col)
            expected_output_values = [[0], [1], [1], [1], [0], [0], [0], [1], [1], [1], [1], [0], [0], [0], [1], [0]]
            actual_output_values = y.tolist()[:16]
            self.assertEqual(expected_output_values, actual_output_values)

    def test__split_dataset_(self):
        file_name = "training_data.csv"
        test_file_path = '../resources/' + file_name
        with open(test_file_path, 'rb') as test_file:
            test_dataset = pd.read_csv(test_file_path)
            processor = DatasetProcessor(test_dataset)
            tr_df, ts_df = processor.__split_dataset__(processor.dataset, 0.8)
            self.assertEqual(38, len(tr_df))
            self.assertEqual(9, len(ts_df))