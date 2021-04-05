from tensorflow import keras
from tensorflow.python.ops import variable_scope

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


def pinch(x, shape, name='', use_bias=False):
  """
  Pinch dim-2 tensor to arbitrary shape.

  use_bias: (Unused).
  """
  if len(x.shape) != 2:
    raise ValueError('pinch shape must be rank 2.')
  first_dim = shape[0]
  second_dim = shape[1]

  scope = variable_scope.get_variable_scope()
  if scope.name:
    op_name = scope.name + '/pinch'
  else:
    op_name = 'pinch'

  with tf.name_scope(scope_name, 'pinch', []) as s:
    full_name = s + '/' + op_name + '/left'
    left_var = tf.get_variable(
        full_name,
        shape=(first_dim, x.shape[0]),
    )

    full_name = s + '/' + op_name + '/right'
    right_var = tf.get_variable(
        full_name,
        shape=(x.shape[0], second_dim),
    )
    
    return tf.matmul(tf.matmul(left_var, x), right_var)


# TODO
def conv_layer(x):
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
