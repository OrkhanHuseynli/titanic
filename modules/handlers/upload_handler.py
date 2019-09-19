from abc import ABC

import tornado.ioloop
from tornado.web import RequestHandler


class UploadHandler(RequestHandler):
    def get(self):
        self.write({'message': 'file'})
