import os
from tempfile import gettempdir


class Utils:
    @staticmethod
    def create_dir(directory):
        tmp = os.path.join(gettempdir(), directory)
        if not os.path.exists(tmp):
            os.makedirs(tmp)