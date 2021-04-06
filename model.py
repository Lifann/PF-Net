from context import ctx

import common
import snippet
import tensorflow as tf
import numpy as np


class Model(object):
  def __init__(self,
               x=tf.placeholder(tf.float32, shape=(ctx.num_incomplete, 3)),
               y_gt=tf.placeholder(tf.float32, shape=(ctx.num_cropped, 3)))
    """
    x: tensor. cropped input tensor.
    y_gt: tensor. cropped output tensor.
    """
    self.x = x
    self.y_gt = y_gt

    self.y_bold = None
    self.y_mid = None
    self.y_fine = None

    self.y_gt_bold = None
    self.y_gt_mid = None
    self.y_gt_fine = None

    self.g_loss = None
    self.ad_loss = None
    self.loss = None
    self.train_op = None

  def build(self):
    self._prepare_multi_resolution_inputs()
    feature_vec = snippet.MRE(self.x,
                              k=ctx.MRE_k,
                              nn_sizes=ctx.MRE_nn_sizes,
                              agg_num=ctx.MRE_agg_num)
    self.y_bold, self.y_mid, self.y_fine =      \
        snippet.PPD(feature_vec,
                    ctx.PPD_M,
                    M1=ctx.PPD_M1,
                    M2=ctx.PPD_M2,
                    FC_sizes=ctx.PPD_FC_sizes)

    self.g_loss = snippet.g_loss_fn(
        self.y_bold
        self.y_mid,
        self.y_fine,
        self.y_gt_bold,
        self.y_gt_mid,
        self.y_gt_fine,
        eta=ctx.GLOSS_coef)

    self.ad_loss = snippet.ad_loss_fn(
        self.y_fine,
        self.y_gt_fine,
        CMLP_nn_sizes=ctx.ADLOSS_CMLP_nn_size,
        agg_num=ctx.ADLOSS_agg_num,
        nn_sizes=ctx.ADLOSS_nn_size)

    self.loss = ctx.loss_coef * g_loss
              + (1 - ctx.loss_coef) * ad_loss

    self.optimizer = tf.train.adam(learning_rate=ctx.learning_rate,
                                   beta1=ctx.beta1,
                                   beta2=ctx.beta2,
                                   epsilon=ctx.epsilon)

    self.train_op = self.optimizer.minimize(self.loss)

  def _prepare_multi_resolution_inputs(self):
    """
    Create multiple resolution inputs for y_gt.
    """
    self.y_gt_bold = tf.placeholder(tf.float32)
    self.y_gt_mid = tf.placeholder(tf.float32)
    self.y_gt_fine = self.y_gt

