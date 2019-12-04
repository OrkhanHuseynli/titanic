import json
from http import HTTPStatus

from tornado.testing import AsyncHTTPTestCase

import main
from modules.core.utils import Utils


class TestApp(AsyncHTTPTestCase):
    def get_app(self):
        return main.make_app()

    def test_upload(self):
        # create a boundary
        boundary = 'SomeRandomBoundary'
        # set the Content-Type header
        headers = {
            'Content-Type': 'multipart/form-data; boundary=%s' % boundary
        }

        response = self.fetch('/api/v1/upload', method='POST', headers=headers, body=self.return_body(boundary))
        self.assertEqual(200, response.code)
        file_name = Utils.get_hashed_name("titanic.csv")
        file_name_with_ext = file_name + ".csv"
        file = Utils.read_file(file_name_with_ext, Utils.UPLOADS_FOLDER)
        expected_file_path = Utils.get_stored_file_path("titanic.csv", Utils.UPLOADS_FOLDER)
        self.assertEqual(expected_file_path, file.name)
        file.close()
        Utils.___remove_file___(expected_file_path)

    def test_train_for_error(self):
        headers = {
            'Content-Type': 'application/json'
        }

        response = self.fetch('/api/v1/train?filename=test.csv', method='POST', body='{}', headers=headers)
        self.assertEqual(HTTPStatus.NOT_FOUND, response.code)

    def test_train(self):
        file_name = "titanic.csv"
        file_name_hashed = Utils.get_hashed_name("titanic.csv")
        file_name_with_ext = file_name_hashed + ".csv"
        with open('resources/titanic.csv', 'rb') as test_file:
            Utils.write_file(file_name_with_ext, test_file.read(), Utils.UPLOADS_FOLDER)
            headers = {
                'Content-Type': 'application/json'
            }
            response = self.fetch('/api/v1/train?filename=' + file_name, method='POST', body='{}', headers=headers)
            self.assertEqual(200, response.code)
            expected = ([[1.0, 1.0],
                         [0.9642857142857143, 0.7241379310344828],
                         [0.8571428571428571, 0.41379310344827586],
                         [0.8392857142857143, 0.3103448275862069],
                         [0.8214285714285714, 0.22988505747126436],
                         [0.75, 0.16091954022988506],
                         [0.6428571428571429, 0.10344827586206896],
                         [0.5535714285714286, 0.011494252873563218],
                         [0.4642857142857143, 0.0],
                         [0.23214285714285715, 0.0],
                         [0.0, 0.0]],
                        [[[56, 87], [0, 0]],
                         [[54, 63], [24, 2]],
                         [[48, 36], [51, 8]],
                         [[47, 27], [60, 9]],
                         [[46, 20], [67, 10]],
                         [[42, 14], [73, 14]],
                         [[36, 9], [78, 20]],
                         [[31, 1], [86, 25]],
                         [[26, 0], [87, 30]],
                         [[13, 0], [87, 43]],
                         [[0, 0], [87, 56]]])
            self.assertEqual(expected[0], json.loads(response.body)['rocList'])
            self.assertEqual(expected[1], json.loads(response.body)['confMatrixList'])
            expected_file_path = Utils.get_stored_file_path(file_name_with_ext, Utils.UPLOADS_FOLDER)
            Utils.___remove_file___(expected_file_path)

    @staticmethod
    def return_body(boundary):
        # create the body
        # opening boundary
        body = '--%s\r\n' % boundary

        # data for field1
        body += 'Content-Disposition: form-data; name="field1"\r\n'
        body += '\r\n'  # blank line
        body += 'Hello\r\n'

        # separator boundary
        body += '--%s\r\n' % boundary

        # data for field2
        body += 'Content-Disposition: form-data; name="filearg"; filename="titanic.csv"\r\n'
        # body += '\r\n'
        body += 'Content-Type: text/csv\r\n\r\n\r\n'
        with open('resources/titanic.csv', 'r') as f:
            body += '%s\r\n' % f.read()
        # the closing boundary
        body += "--%s--\r\n" % boundary
        return body
