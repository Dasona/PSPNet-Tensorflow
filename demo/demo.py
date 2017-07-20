import os
import sys
import time
from StringIO import StringIO

import argparse as configargparse
import cv2
import logging
import numpy as np
import scipy
import scipy.misc as misc
import tensorflow as tf
import tornado
import tornado.httpserver as httpserver
import tornado.web as web
import urllib
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors

from PIL import Image


app = None # pylint: disable=invalid-name

FLAGS = tf.app.flags.FLAGS

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94



palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
           (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
           (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
           (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
           (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
           (0.0, 0.25, 0.5)]

my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, 21)


try:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()


def setup_log(dest=None):
    root = logging.getLogger("")
    formatter = logging.Formatter("%(asctime)s %(message)s", "%m/%d %H:%M:%ST%z")
    root.handlers = []

    handlers = [logging.StreamHandler(stream=sys.stdout)]
    if dest:
        handlers.append(logging.FileHandler(dest))

    for handler in handlers:
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    root.setLevel(logging.INFO)


def _read_image(data):
  return misc.imread(StringIO(data))


def _download_image(url):
  image = urllib.urlopen(url)
  data = StringIO()
  while True:
    buf = image.read(10000000)
    if len(buf) == 0:
      break
    data.write(buf)

  logging.info('{} bytes read'.format(len(buf)))

  return misc.imread(data)


class EchoHandler(web.RequestHandler):
  def initialize(self, app):
    self._app = app

  def get(self, texts):
    self.write(self._app.echo(texts))


class TestUIHandler(web.RequestHandler):
  def initialize(self, app):
    self._app = app


  def get(self):
    self.render("template.html", title="Semantic Segmentation Workbench")


  def post(self):
    req = self.request

    if 'url' in req.body_arguments:
      logging.info('url is in argument {}'.format(req.body_arguments['url']))
      image = _download_image(req.body_arguments['url'][0])
    else:
      logging.info('url is not in argument')
      image = _read_image(req.files['image'][0]['body'])

    H, W = image.shape[0], image.shape[1]

    resized_image = scipy.misc.imresize(image, (512, 473))

    label = self._app.predict(resized_image)

    label = misc.imresize(label, (H, W), interp='nearest')
    fig = plt.figure()
    ax = fig.add_subplot('121')
    ax.imshow(image)
    ax = fig.add_subplot('122')
    ax.matshow(label, vmin=0, vmax=21, cmap=my_cmap)
    output = StringIO()
    plt.savefig(output)
    self.write(output.getvalue())
    self.set_header('Content-type', 'JPG')



class App(object):
  def __init__(self, config):

    self._graph = tf.Graph()

    tf_config = None
    if not config.get("gpu", None):
        tf_config = tf.ConfigProto(device_count={"GPU":0})
    else:
        tf_config = tf.ConfigProto(device_count={"GPU":1})
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.per_process_gpu_memory_fraction=config["gpu_memory_fraction"]

    self._sess = tf.Session(config=tf_config, graph=self._graph)

    with self._sess.graph.as_default():
        graph_def = tf.GraphDef()
        with open(config['model'], 'rb') as file:
            graph_def.ParseFromString(file.read())
        tf.import_graph_def(graph_def, name="")

    self._input_x = self._sess.graph.get_operation_by_name('ph_input_x').outputs[0]
    self._pred = self._sess.graph.get_operation_by_name('predictions').outputs[0]
    self._softmax = self._sess.graph.get_operation_by_name('softmax').outputs[0]

    self._http_app = web.Application(
        handlers=[
            web.URLSpec(r"/api/echo/(.*)", EchoHandler, dict(app=self)),
            web.URLSpec(r"/api/image", EchoHandler, dict(app=self)),
            web.URLSpec(r"/ui/segmentation", TestUIHandler, dict(app=self))
        ],
        debug=config["debug"],
    )


  def http_app(self):
    return self._http_app


  def predict(self, image):
    H, W = image.shape[0], image.shape[1]
    assert H == 512
    assert W == 473

    before = time.time()
    result = self._sess.run(self._pred, feed_dict={self._input_x: image})[0]
    logging.debug('take {} ms'.format(time.time() - before))
    return result


def main():
  """main"""
  parser = configargparse.ArgumentParser()

  parser.add_argument("--debug", dest="debug", default=False, action="store_true")
  parser.add_argument("--no-debug", dest="debug", action="store_false")
  parser.add_argument("--log", dest="log", default="")

  parser.add_argument("--host", dest="host", default=os.environ.get("BIND", "127.0.0.1"))
  parser.add_argument("--port", dest="port", type=int, default=int(os.environ.get("PORT", 8000)))

  parser.add_argument("--model", dest="model", required=True)

  parser.add_argument("--gpu", dest="gpu", default=True, action="store_true")
  parser.add_argument("--no-gpu", dest="gpu", action="store_false")
  parser.add_argument("--gpu-memory-fraction", type=float, default=0.40, dest="gpu_memory_fraction")

  config = vars(parser.parse_args())
  setup_log(config["log"])

  logging.info("config: %s", config)

  app = App(config)

  server = httpserver.HTTPServer(app.http_app())
  server.bind(config["port"], address=config["host"])
  server.start()
  logging.info("Server Start! Listen: %s", [x.getsockname() for x in server._sockets.values()]) # pylint: disable=protected-access
  tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
