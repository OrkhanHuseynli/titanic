import tornado
from tornado.ioloop import IOLoop
from tornado.options import define, options
from tornado.web import Application

from modules.handlers.training_handler import TrainingHandler
from modules.handlers.upload_handler import UploadHandler

define('port', default=8888, help='port to listen on')


def make_app():
    return tornado.web.Application([
        (r"/api/v1/upload", UploadHandler),
        (r"/api/v1/train", TrainingHandler)
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(options.port)
    print('Listening on http://localhost:%i' % options.port)
    tornado.ioloop.IOLoop.current().start()
