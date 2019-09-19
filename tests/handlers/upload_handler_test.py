import unittest
from tornado.testing import AsyncTestCase, gen_test
from tornado.web import Application
from tornado.httpserver import HTTPRequest
from unittest.mock import Mock

class TestUploadHandler(unittest.TestCase):
    @gen_test
    def test_no_from_date_param(self):
        print("test")
        # mock_application = Mock(spec=Application)
        # payload_request = HTTPRequest(
        #     method='GET', uri='/upload', headers=None, body=None
        # )
        # handler = UploadHandler(mock_applciation, payload_request)
        # with self.assertRaises(ValueError):
        #     yield handler.get()
