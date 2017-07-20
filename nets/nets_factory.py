from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from nets import pspnet_rgbd,pspnet_rgb

slim = tf.contrib.slim

networks_map = {'pspnet_rgbd': pspnet_rgbd.pspnet_v1_50,
                'pspnet_rgb' : pspnet_rgb.pspnet_v1_50,
               }

arg_scopes_map = {'pspnet_rgbd': pspnet_rgbd.pspnet_arg_scope,
                  'pspnet_rgb' : pspnet_rgb.pspnet_arg_scope,
                 }


def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False, **kwargs):
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)

  arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
  func = networks_map[name]

  @functools.wraps(func)
  def network_fn(images, **kwargs):
    with slim.arg_scope(arg_scope):
      return func(inputs=images, num_classes=num_classes, is_training=is_training, **kwargs)
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn
