from tornado.testing import AsyncHTTPTestCase

import main
class TestHelloApp(AsyncHTTPTestCase):
    def get_app(self):
        return main.make_app()

    def test_homepage(self):
        response = self.fetch('/')
        self.assertEqual(200, response.code)
        self.assertEqual('{"message": "file"}', response.body.decode('utf-8'))