from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

slim = tf.contrib.slim

_FILE_PATTERN = 'data_%s_*.tfrecord'

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A annotation image of varying size.',
    'depth': 'A HHA Depth image for corresponding image'
}


def get_split(split_name, training_size, validation_size, num_classes, dataset_dir, file_pattern=None, reader=None):
  SPLITS_TO_SIZES = {'training': training_size, 'validation': validation_size}
  _NUM_CLASSES = num_classes
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'label/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'label/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'depth/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'depth/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Image('label/encoded', 'label/format',
                                            channels=1),
      'depth': slim.tfexample_decoder.Image('depth/encoded', 'depth/format',
                                            channels=3)
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES)
