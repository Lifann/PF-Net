"""
snippet.py is used to abstract blocks in whole model,
for better readability.
"""

from context import ctx

import common
import tensorflow as tf
import utils


def CMLP(xyz, nn_sizes, name='CMLP', use_bn=False,
         activation=None, agg_num=0, use_legacy=False,
         **kwargs):
  """
  Get latent vector from point cloud coordinates.

  Args:
    xyz: point cloud data tensor, with shape (N, 3) or (3, N)
    nn_sizes: list of integers. Size of each layers.
    use_bn: bool. Whether use batch normalization.
    activation: activation layer. None do not use.
    agg_num: number of aggregation layers count from layers on tail.
    use_legacy: wether if use legacy mode run, which enable sharing
      on variables.

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
    layer_name = name + '_' + str(i) + '_' +str(size)
    x = xyz
    x = common.dense_layer(x,
                           size,
                           name=layer_name,
                           use_bias=True,
                           use_bn=use_bn,
                           activation=activation,
                           is_training=ctx.is_training,
                           use_legacy=use_legacy,
                           **kwargs)  # (3, U)
    if i < len(nn_sizes) - agg_num:
      continue
    else:
      # Record tensor for concat
      x = tf.reshape(x, [1, 1] + list(x.shape))
      x = tf.nn.max_pool(x, ctx.max_pool_kernel, ctx.max_pool_stride, padding='VALID')
      x = tf.reshape(x, (1, -1))
      out.append(x)
  latent_feature = tf.concat(out, axis=1)  # (1, R)
  tensor_size = utils.size_from_shape(latent_feature.shape)
  latent_feature = tf.reshape(latent_feature, (tensor_size, 1))

  return latent_feature  # (R, 1)


def MRE(cloud_data, k=2, nn_sizes=[], agg_num=4):
  """
  Multi-resolution Encoder(MRE).

  Args:
    cloud_data: data in PointCloud object which has been cropped.

  Returns:
    A tensor vector.
  """
  detail_data = cloud_data

  num_points = int(detail_data.shape.as_list()[0] / k)
  secondary_data = common.randomly_down_sample(detail_data, num_points)

  num_points = int(num_points / k)
  primary_data = common.randomly_down_sample(secondary_data, num_points)

  datas = [detail_data, secondary_data, primary_data]

  final_latent_map = []
  for x in datas:
    latent_vec = CMLP(
        x, nn_sizes, use_bn=True, activation='relu', agg_num=agg_num)  # (R, 1)
    final_latent_map.append(latent_vec)
  final_latent_map = tf.concat(final_latent_map, axis=1)  # (R, 3)

  final_feature_vec = common.dense_layer(final_latent_map,
                                         1,
                                         use_bias=True,
                                         use_bn=True,
                                         is_training=ctx.is_training,
                                         activation='relu')  # (R, 1)
  final_feature_vec = common.dense_layer(final_latent_map,
                                         1,
                                         use_bias=True,
                                         use_bn=True,
                                         is_training=ctx.is_training,
                                         activation='linear')  # (R, 1)

  return final_feature_vec


# TODO
def PPD(feature_vec,
        M,
        M1=64,
        M2=128,
        FC_sizes=[1024, 512, 256],
  ):
  """
  Point Pyramid Decoder.

  Args:
    feature_vec: input feature tensor vector, with shape (R, 1), whom
      generated from MRE.
    M: detail branch size.
    M1: primary branch size.
    M2: secondary branch size.
    FC_sizes: output sizes of FC layers. It's length must be 3.
  """
  feature_vec = tf.reshape(feature_vec, (1, -1))
  detail_fc = common.dense_layer(feature_vec,
                                 FC_sizes[0],
                                 use_bias=True,
                                 use_bn=False,
                                 is_training=ctx.is_training,
                                 activation='relu')
  detail_map = common.pinch_vec(detail_fc, M * M2)
  detail_map = tf.reshape(detail_map, (M, M2))
  detail_map = common.conv_layer(detail_map, M2, M,
                                 kernel_size=3,
                                 expand_dim=True,
                                 activation='relu')
  detail_map = tf.reshape(detail_map, (M2, int(M / M2), 3))

  secondary_fc = common.dense_layer(feature_vec,
                                    FC_sizes[1],
                                    use_bias=True,
                                    use_bn=False,
                                    is_training=ctx.is_training,
                                    activation='relu')
  secondary_map = common.pinch_vec(secondary_fc, M1 * M2)
  secondary_map = tf.reshape(secondary_map, (M1, M2))
  secondary_map = common.conv_layer(secondary_map, M1, M2,
                                    kernel_size=3,
                                    expand_dim=True,
                                    activation='relu')
  secondary_map = tf.reshape(secondary_map, (M2, int(M2 / M1), 3))

  primary_fc = common.dense_layer(feature_vec,
                                  FC_sizes[1],
                                  use_bias=True,
                                  use_bn=False,
                                  is_training=ctx.is_training,
                                  activation='relu')
  primary_map = common.pinch_vec(primary_fc, 3 * M1)
  primary_map = tf.reshape(primary_map, (M1, 3))
  primary_map = common.conv_layer(primary_map, M1, 3,
                                  kernel_size=3,
                                  expand_dim=False,  # primary branch do not expand_dim.
                                  activation='relu')
  primary_out = primary_map

  secondary_out = tf.reshape(
      tf.expand_dims(primary_out) + secondary_map,
      (M2, 3))
  detail_out = tf.reshape(
      tf.expand_dims(secondary_out) + detail_map,
      (M, 3))

  return primary_out, secondary_out, detail_out


def g_loss_fn(y_bold, y_mid, y_fine,
              y_gt_bold, y_gt_mid, y_gt_fine,
              eta=0.001):
  sub_bold = y_bold - y_gt_bold
  dsit_bold = tf.reduce_sum(sub_bold * sub_bold, axis=1)

  sub_mid = y_mid - y_gt_mid
  dsit_mid = tf.reduce_sum(sub_mid * sub_mid, axis=1)

  sub_fine = y_fine - y_gt_fine
  dsit_fine = tf.reduce_sum(sub_fine * sub_fine, axis=1)

  loss = dist_fine            \
       + eta * dist_mid       \
       + 2 * eta * dist_bold
  return loss


# TODO: Must share variables
def ad_loss_fn(y, y_gt,
              CMLP_nn_sizes=[64, 64, 128, 256],
              agg_num=3,
              nn_sizes=[256, 128, 16, 1]):
  def struct_fn(x):
    ts = CMLP(x, CMLP_nn_sizes, name='ad_loss_CMLP', use_bn=True,
              activation='relu', agg_num=CMLP_agg_num,
              use_legacy=True)  # (R, 1)
    for i, size in enumerate(nn_size):
      layer_name = 'g_loss_linear/' + str(i) + '_' + str(size)
      ts = common.dense_layer(ts, size, name=layer_name, use_bias=True,
                              use_bn=True, is_training=ctx.is_training,
                              activation='relu', use_legacy=True)
    return ts

  with tf.variable_scope('g_loss', default_name='g_loss',
                         reuse=tf.AUTO_REUSE):
    logit = struct_fn(y)
    logit_gt = struct_fn(y_gt)

    ad_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=tf.reshape(logit, (-1, 1)),
        labels=tf.reshape(logit_gt, (-1, 1)),
    )
    return ad_loss
