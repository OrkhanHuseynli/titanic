import os
from unittest import TestCase

from modules.core.utils import Utils


class TestUtils(TestCase):
    def testCreateDir(self):
        with open('../resources/titanic.csv', 'rb') as test_file:
            file = Utils.write_file("test.csv", test_file.read(), Utils.UPLOADS_FOLDER)
            with open(file.name, "r") as reader:
                first_line = reader.readline()
                with open('../resources/titanic.csv', 'r') as test_file_reader:
                    self.assertEqual(test_file_reader.readline(), first_line)
            Utils.remove_file(file.name)
