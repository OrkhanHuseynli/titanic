import os
import uuid
from abc import ABC

import tornado.ioloop
from tornado.web import RequestHandler

__UPLOADS__ = "uploads/"

class UploadHandler(RequestHandler):
    def get(self):
        self.write({'message': 'file'})

    def post(self):
        file_info = self.request.files['filearg'][0]
        file_name = file_info['filename']
        extn = os.path.splitext(file_name)[1]
        cname = str(uuid.uuid4()) + extn
        fh = open(__UPLOADS__ + cname, 'w')
        fh.write(file_info['body'])
        self.finish(cname + " is uploaded!! Check %s folder" % __UPLOADS__)