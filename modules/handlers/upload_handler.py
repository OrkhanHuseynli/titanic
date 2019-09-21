import os
import uuid
from abc import ABC

import tornado.ioloop
from tornado.web import RequestHandler

from modules.core.utils import Utils

class UploadHandler(RequestHandler):
    def get(self):
        self.write({'message': 'file'})

    def post(self):
        file_info = self.request.files['filearg'][0]
        file_name = file_info['filename']
        extn = os.path.splitext(file_name)[1]
        cname = str(uuid.uuid4()) + extn
        Utils.write_file(cname, file_info['body'], Utils.UPLOADS_FOLDER)
        self.finish(cname + " is uploaded!! Check %s folder" % Utils.UPLOADS_FOLDER)