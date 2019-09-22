import os
from unittest import TestCase

import pandas as pd

from modules.core.utils import Utils
from tempfile import gettempdir


class TestUtils(TestCase):
    def test_create_dir(self):
        file_name = "test.csv"
        with open('../resources/titanic.csv', 'rb') as test_file:
            file = Utils.write_file(file_name, test_file.read(), Utils.UPLOADS_FOLDER)
            with open(file.name, "r") as reader:
                first_line = reader.readline()
                with open('../resources/titanic.csv', 'r') as test_file_reader:
                    self.assertEqual(test_file_reader.readline(), first_line)
                file = Utils.read_file(file_name, Utils.UPLOADS_FOLDER)
                self.assertEqual(file.name, os.path.join(gettempdir(), Utils.UPLOADS_FOLDER, file_name))
                file.close()
            Utils.___remove_file___(file.name)

    def test_read_dataset(self):
        file_name = "titanic.csv"
        test_file_path = '../resources/' + file_name
        with open(test_file_path, 'rb') as test_file:
            test_dataset = pd.read_csv(test_file_path)
            hashed_file_name = Utils.get_hashed_name(file_name) + ".csv"
            Utils.write_file(hashed_file_name, test_file.read(), Utils.UPLOADS_FOLDER)
            dataset = Utils.read_dataset(file_name)
            self.assertEqual(test_dataset.shape, dataset.shape)
            Utils.remove_file(hashed_file_name, Utils.UPLOADS_FOLDER)
