from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from preprocessing import rgb_preprocessing, rgbd_preprocessing

slim = tf.contrib.slim


def get_preprocessing(name, is_training=False):
  preprocessing_fn_map = {
      'rgb': rgb_preprocessing,
      'rgbd': rgbd_preprocessing,
  }

  if name not in preprocessing_fn_map:
    raise ValueError('Preprocessing name [%s] was not recognized' % name)

  def preprocessing_fn(image, output_height, output_width, **kwargs):
    return preprocessing_fn_map[name].preprocess_image(
        image, output_height, output_width, is_training=is_training, **kwargs)

  return preprocessing_fn
