import os
from unittest import TestCase

from modules.core.utils import Utils
from tempfile import gettempdir


class TestUtils(TestCase):
    def testCreateDir(self):
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
