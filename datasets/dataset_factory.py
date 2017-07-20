from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import rgb,rgbd

datasets_map = {
    'rgb': rgb,
    'rgbd': rgbd,
}


def get_dataset(name, split_name, training_size, validation_size, num_classes, dataset_dir, file_pattern=None, reader=None):
  if name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % name)

  return datasets_map[name].get_split(
      split_name,
      training_size,
      validation_size,
      num_classes,
      dataset_dir,
      file_pattern,
      reader)
