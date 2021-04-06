from tensorflow import keras
from tensorflow import initializers
from tensorflow.python.ops import variable_scope

from context import ctx

import tensorflow as tf


def dense_layer(x,
                size,
                name='dense',
                use_bias=False,
                use_bn=False,
                is_training=True,
                activation=None,
                use_legacy=False,
                **kwargs):
  if use_legacy:
    return _legacy_dense_layer(x, size, name=name, use_bn=use_bn,
                               is_training=is_training, activation=activation)

  net = keras.layers.Dense(size, name=name, activation=activation,
                           use_bias=use_bias, **kwargs)
  x = net(x)
  if use_bn:
    bn = keras.layers.BatchNormalization(name=name + '/BN')
    x = bn(x, training=is_training, **kwargs)
  return x


def _legacy_dense_layer(x, size, name='legacy_dense',
                        use_bn='relu',
                        is_training=True,
                        activation='relu'):
  shape = (x.shape.as_list()[-1], size)
  var =tf.get_variable(
      name, dtype=tf.float32, shape=shape,
      initializer=initializers.truncated_normal())
  ts = tf.matmul(x, var)
  if use_bn:
    ts = tf.layers.batch_normalization(ts, training=is_training)
  if activation:
    act_fn = tf.nn.relu
    try:
      act_fn = getattr(tf.nn, activation)
    except:
      raise KeyError('tf.nn has no module: {}'.format(activation))
    ts = act_fn(ts)
  return ts


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
    scope_name = scope.name + '/pinch'
  else:
    scope_name = 'pinch'

  with tf.name_scope(scope_name, 'pinch', []) as s:
    full_name = s + '/' + '/left'
    left_var = tf.get_variable(
        full_name,
        shape=(first_dim, x.shape[0]),
    )

    full_name = s + '/' + '/right'
    right_var = tf.get_variable(
        full_name,
        shape=(x.shape[0], second_dim),
    )
    
    return tf.matmul(tf.matmul(left_var, x), right_var)


def pinch_vec(x, size):
  """
  Pinch vec-(a, 1) to vec-(size, 1)
  """
  if len(x.shape) != 2:
    raise ValueError('pinch shape must be rank 2.')
  if x.shape[0] != 1:
    raise ValueError('shape[1] must be 1.')
  
  scope = variable_scope.get_variable_scope()
  if scope.name:
    scope_name = scope.name + '/pinch_vec'
  else:
    scope_name = 'pinch_vec'

  with tf.name_scope(scope_name, 'pinch_vec', []) as s:
    full_name = s + '/' + '/right'
    right_var = tf.get_variable(
        full_name,
        shape=(x.shape[1], size),
    )
    return tf.matmul(x, right_var)


def get_stride(C, W, K):
  return max(int((W - K) / (C - 1)), 1)


def conv_layer(x, M1, M2,
               kernel_size=3, expand_dim=False,
               activation=None, padding='valid'):
  """
  filter = M1
  kernel = arbitrary value.
  int(M2 / stride) == 3 * M2 / M1 == 6     (1)
  int(M2 / stride) == 3                    (2)

  Args:
    M2: first_dim of input.
    M1: second_dim of input.
  """
  x = tf.reshape(x, (M2, M1))
  if expand_dim:
    target_dim = int(3 * M2 / M1)
  else:
    target_dim = 3

  stride = get_stride(target_dim, M2, kernel_size)
  x = tf.expand_dims(x, 0)
  net = keras.layers.Conv1D(M1,
                            kernel_size,
                            stride,
                            activation=activation,
                            padding=padding)
  conv_tensor = net(x)  # (1, 3 * M2 / M1, M1)
  conv_tensor = tf.reshape(conv_tensor, (target_dim, M1))
  return conv_tensor


def randomly_down_sample(x, num_points):
  if (len(x.shape)) != 2:
    raise ValueError('x must be rank 2.')
  if x.shape[1] != 3:
    raise ValueError('Unmatched shape.')

  size = x.shape.as_list()[0]
  if num_points > size:
    raise ValueError('Try sample {} points from {} points'.format(
                     num_points, x.shape[0]))

  indices = tf.random.uniform((num_points, ), minval=0, maxval=size, dtype=tf.int32)
  return tf.gather(x, indices)


def get_multi_resolution_clouds(cloudx):
  size = cloud.length
  bold_cloud = cloudx.down_sample(ctx.PPD_M1)
  mid_cloud = cloudx.down_sample(ctx.PPD_M2)
  fine_cloud = cloudx
  return (bold_cloud, mid_cloud, fine_cloud)

