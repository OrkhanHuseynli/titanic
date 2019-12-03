import os
from http import HTTPStatus

import pandas as pd
from tornado.web import RequestHandler, HTTPError

from modules.core.utils import Utils
from modules.services.log_training_service import LogTrainingService


class TrainingHandler(RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "content-type")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    def post(self):
        file_name=self.get_argument("filename", None, True)
        extn = os.path.splitext(file_name)[1]
        cname = Utils.get_hashed_name(file_name) + extn
        alpha = 0.01
        iterations = 10000
        testing_size = 0.8
        tuple_of_X_col, y_column_index = (3, 4, 5, 6, 7), 2
        service = LogTrainingService()
        try:
            roc_list, conf_matrix_list, thresholds_list = service.train(cname, testing_size, alpha, iterations, tuple_of_X_col, y_column_index)
        except OSError:
            raise HTTPError(HTTPStatus.NOT_FOUND, 'Corresponding file does not exist for training : ' + file_name)
        operation_info = {'operation': 'training',
                          'fileName': file_name,
                          'rocList': roc_list,
                          'confMatrixList': conf_matrix_list,
                          'thresholdList': thresholds_list}
        self.finish(operation_info)
