"""
snippet.py is used to abstract blocks in whole model,
for better readability.
"""

import common


def CMLP(xyz, nn_sizes, use_bn=False, activation=None, agg_num=0, **kwargs):
  """
  Get latent vector from point cloud coordinates.

  Args:
    xyz: point cloud data tensor, with shape (N, 3) or (3, N)
    use_bn: bool. Whether use batch normalization.
    activation: activation layer. None do not use.
    agg_num: number of aggregation layers count from layers on tail.

  Returns:
    An aggregated tensor with (3, C) shape, where C = sum<agg>(nn_sizes).
  """
  if xyz.shape[1] == 3 and len(xyz.shape) == 2:
    xyz = tf.transpose(xyz)
  elif xyz.shape[0] == 3 and len(xyz.shape) == 2:
    pass
  else:
    raise ValueError('point cloud data got wrong shape.')

  out = []
  for i, size in enumerate(nn_sizes):
    x = common.dense_layer(x,
                           size,
                           use_bn=use_bn,
                           activation=activation,
                           is_training=FLAGS.is_training,
                           **kwargs)
    if i < len(nn_sizes) - agg_num:
      continue
    else:
      # Record tensor for concat
      x = tf.reshape(x, [1, 1] + list(x.shape))
      x = tf.nn.max_pool(x, (1, 1, 3, 1), 'VALID')
      out.append(x)
  latent_feature = tf.concat(out, axis=1)  # (1, R)
  tensor_size = utils.size_from_shape(latent_feature.shape)
  latent_feature = tf.reshape(latent_feature, (tensor_size, 1))

  return latent_feature  # (R, 1)


def MRE(incomplete_cloud, k=2, nn_sizes=[], agg_num=4):
  """
  Multi-resolution Encoder(MRE).

  Args:
    incomplete_cloud: A PointCloud object which has been cropped.

  Returns:
    A tensor vector.
  """
  detail_data = cloud.tf_data

  num_points = int(detail_data.shape[0] / k)
  secondary_data = cloud.tf_down_sample(num_points)

  num_points = int(num_points / k)
  primary_data = common.randomly_down_sample(secondary_data, num_points)

  datas = [detail_data, secondary_data, primary_data]

  final_latent_map = []
  for x in datas:
    latent_vec = CMLP(
        x, nn_sizes, use_bn=True, activation='relu', agg_num=agg_num)  # (R, 1)
    final_latent_map.append(latent_vec)
  final_latent_map = tf.concat(final_latent_map, axis=1)  # (R, 3)

  final_feature_vec = dense_layer(final_latent_map,
                                  1,
                                  use_bn=True,
                                  is_training=FLAGS.is_training,
                                  activation='relu')  # (R, 1)
  final_feature_vec = dense_layer(final_latent_map,
                                  1,
                                  use_bn=True,
                                  is_training=FLAGS.is_training,
                                  activation='linear')  # (R, 1)

  return final_feature_vec

  #tensor_size = utils.size_from_shape(final_feature_vec.shape)
  #return tf.transpose(final_feature_vec)  # (1, R)


# TODO
def PPD(final_feature_vec)
  pass
