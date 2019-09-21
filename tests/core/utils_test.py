import os
from unittest import TestCase

from modules.core.utils import Utils


class TestUtils(TestCase):
    def testCreateDir(self):
        file = Utils.write_file("test.csv", "content", Utils.UPLOADS_FOLDER)
        Utils.remove_file(file.name)
