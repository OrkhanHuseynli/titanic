import os
import io

import pandas as pd
from tornado.web import RequestHandler

from modules.core.utils import Utils


class UploadHandler(RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    def post(self):
        file_info = self.request.files['filearg'][0]
        file_name = file_info['filename']
        extn = os.path.splitext(file_name)[1]
        cname = Utils.get_hashed_name(file_name) + extn
        Utils.write_file(cname, file_info['body'], Utils.UPLOADS_FOLDER)
        dataset_info = self._get_data_summary_(file_info['body'])
        operation_info = {'operation': 'file_upload', 'fileName': file_name}
        dataset_info.update(operation_info)
        self.finish(dataset_info)

    def _get_data_summary_(self, data: bytes) -> any:
        dataset = pd.read_csv(io.BytesIO(data))
        return {'dataSize': "{}x{}.".format(dataset.size, len(dataset.columns)), 'columns': list(dataset.columns)}
