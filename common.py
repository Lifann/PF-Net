from tensorflow import keras

import coommon
import tensorflow as tf


def dense_layer(x,
                size,
                use_bn=False,
                is_training=True,
                activation=None,
                **kwargs):
  net = keras.layers.Dense(size, use_bias=True, **kwargs)
  x = net(x)
  if use_bn:
    bn = keras.layers.BatchNormalization()
    x = bn(x, training=is_training, **kwargs)
  return x


def legacy_dense_layer(input_shape,
                       output_shape,


def conv_layer():
  pass


def randomly_down_sample(x, num_points):
  if (x.shape) != 2:
    raise ValueError('x must be rank 2.')
  if x.shape[1] != 3:
    raise ValueError('Unmatched shape.')

  if num_points > size:
    raise ValueError('Try sample {} points from {} points'.format(
                     num_points, x.shape[0]))

  indices = tf.random.uniform((num_points, ), minval=0, maxval=self.length, dtype=tf.int32)
  return tf.gather(x, indices)

