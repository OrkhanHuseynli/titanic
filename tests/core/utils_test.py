from unittest import TestCase

from modules.core.utils import Utils


class TestUtils(TestCase):
    def testCreateDir(self):
        Utils.create_dir("uploads")