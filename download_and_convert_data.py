from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import download_and_convert_rgb,download_and_convert_rgbd

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name', None,
    'The name of the dataset to convert, one of "cifar10", "flowers", "mnist".')

tf.app.flags.DEFINE_string(
    'dataset_dir', None,
    'The directory where the output TFRecords and temporary files are saved.')

tf.app.flags.DEFINE_string(
    'dataset_type', None,
    'The type of images ( RGB or RGBD ) images present in the database')


def main(_):
  if not FLAGS.dataset_name:
    raise ValueError('You must supply the dataset name with --dataset_name')
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  if not FLAGS.dataset_type:
    raise ValueError('You must supply the dataset image type with --dataset_type')
  if FLAGS.dataset_type == 'rgb':
    download_and_convert_rgb.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_type == 'rgbd':
    download_and_convert_rgbd.run(FLAGS.dataset_dir)
  else:
    raise ValueError(
        'dataset_type [%s] was not recognized.' % FLAGS.dataset_dir)
if __name__ == '__main__':
  tf.app.run()

