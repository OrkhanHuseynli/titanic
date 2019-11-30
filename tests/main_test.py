from http import HTTPStatus

from tornado.testing import AsyncHTTPTestCase

from modules import main
from modules.core.utils import Utils

_UPLOAD_FILE_NAME_ = 'titanic.csv'


class TestApp(AsyncHTTPTestCase):
    def get_app(self):
        return main.make_app()

    # def test_app(self):
    #     response = self.fetch('/uploads')
    #     self.assertEqual(200, response.code)
    #     self.assertEqual('{"message": "file"}', response.body.decode('utf-8'))

    def test_upload(self):
        # create a boundary
        boundary = 'SomeRandomBoundary'
        # set the Content-Type header
        headers = {
            'Content-Type': 'multipart/form-data; boundary=%s' % boundary
        }

        response = self.fetch('/uploads', method='POST', headers=headers, body=self.return_body(boundary))
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

        response = self.fetch('/train?filename=test.csv', method='POST', body='{}', headers=headers)
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
            response = self.fetch('/train?filename=' + file_name, method='POST', body='{}', headers=headers)
            self.assertEqual(200, response.code)
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
