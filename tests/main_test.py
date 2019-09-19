from tornado.testing import AsyncHTTPTestCase

import main
class TestApp(AsyncHTTPTestCase):
    def get_app(self):
        return main.make_app()

    def test_app(self):
        response = self.fetch('/uploads')
        self.assertEqual(200, response.code)
        self.assertEqual('{"message": "file"}', response.body.decode('utf-8'))