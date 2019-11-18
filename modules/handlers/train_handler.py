import os
from tornado.web import RequestHandler

from modules.core.utils import Utils


class TrainHandler(RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    def post(self):
        file_info = self.request
        self.finish({'rocPoints': []})
