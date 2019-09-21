import unittest
from tornado.testing import AsyncTestCase, gen_test
from tornado.web import Application, RequestHandler
from tornado.httpserver import HTTPRequest
from unittest.mock import Mock

from modules.core.utils import Utils
from modules.handlers.upload_handler import UploadHandler


class TestUploadHandler(unittest.TestCase):
    def test_no_from_date_param(self):
        with open('../resources/titanic.csv', 'rb') as test_file:
            print(test_file.name)