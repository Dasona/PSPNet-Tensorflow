#!/usr/bin/env python
import numpy as np
import glob
import tensorflow as tf
import scipy
import scipy.misc as misc
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors

tf.app.flags.DEFINE_string(
    'image_dir_path', None, 'Directory for images to segment')

tf.app.flags.DEFINE_string(
    'dataset_type', 'rgb', 'Type of images to segment')

tf.app.flags.DEFINE_string(
    'output_path', None, 'Output directory where images need to be stored')

tf.app.flags.DEFINE_string(
    'model_path', None, 'Directory for frozen model')

tf.app.flags.DEFINE_string(
    'image_extension', None, 'Extension of image files in the directory')

FLAGS = tf.app.flags.FLAGS

palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
           (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
           (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
           (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
           (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
           (0.0, 0.25, 0.5)]
my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, 21)

def main(_):

  g = tf.Graph()
  sess = tf.Session(graph=g)

  with sess.graph.as_default():
      graph_def = tf.GraphDef()
      with open(FLAGS.model_path, 'rb') as file:
          graph_def.ParseFromString(file.read())
      tf.import_graph_def(graph_def, name="")

  input_x = sess.graph.get_operation_by_name('ph_input_x').outputs[0]
  pred = sess.graph.get_operation_by_name('predictions').outputs[0]
  
  output_path=FLAGS.output_path
  if not tf.gfile.Exists(output_path):
    tf.gfile.MakeDirs(output_path)


  for filename in glob.glob(FLAGS.image_dir_path+'/*.'+FLAGS.image_extension):
    print("\nProcessing : "+str(filename))
    input_image_ori = scipy.misc.imread(filename)
    H, W = input_image_ori.shape[0], input_image_ori.shape[1]

    input_image = scipy.misc.imresize(input_image_ori, (512, 473))

    p = sess.run(pred, feed_dict={input_x: input_image})[0]

    fname = filename[filename.rfind('/'):]
    scipy.misc.imsave(output_path+fname, p)


if __name__ == '__main__':
  tf.app.run()
