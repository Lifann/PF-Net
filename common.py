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


def conv_layer(x, M1, M2,
               kernel_size=3, expand_dim=False,
               activation=None, padding='same'):
  """
  filter = M1
  kernel = arbitrary value.
  int(M2 / stride) == 3 * M2 / M1 == 6     (1)
  int(M2 / stride) == 3                    (2)
  """
  x = tf.reshape(1, M2, M1)
  if expand_dim:
    target_dim = int(3 * M2 / M1)
  else:
    target_dim = 3

  test_strides = [int(M2 / target_dim), int(M2 / target_dim) + 1]
  if int(M2 / test_strides[0]) == target_dim:
    stride = test_strides[0]
  elif int(M2 / test_strides[1]) == target_dim:
    stride = test_strides[0]
  else:
    # This should never happen.
    raise ValueError('Cannot get valid stride for convolution.')
  net = keras.layers.Conv1D(M1,
                            kernel_size,
                            stride,
                            activation=activation,
                            padding=padding)
  conv_tensor = net(x)
  return tf.squeeze(conv_tensor)


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

