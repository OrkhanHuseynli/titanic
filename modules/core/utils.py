import os
from tempfile import gettempdir


class Utils:
    UPLOADS_FOLDER = "TitanicApp/uploads"
    @staticmethod
    def create_dir(directory):
        tmp = os.path.join(gettempdir(), directory)
        if not os.path.exists(tmp):
            os.makedirs(tmp)

    @staticmethod
    def write_file(file_name, content, file_dir):
        Utils.create_dir(file_dir)
        file_path = os.path.join(gettempdir(), file_dir, file_name)
        # if not os.path.exists(file_path):
        with open(file_path, "wb") as file:
            file.write(content)
        return file

    @staticmethod
    def remove_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
