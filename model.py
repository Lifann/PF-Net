from context import ctx

import common
import snippet
import tensorflow as tf
import numpy as np


class Model(object):
  def __init__(self,
               x=tf.placeholder(tf.float32, shape=(ctx.num_incomplete, 3)),
               y_gt=tf.placeholder(tf.float32, shape=(ctx.num_cropped, 3))):
    """
    x: tensor. cropped input tensor.
    y_gt: tensor. ground truth missing part.
    """
    self.x = x
    self.y_gt = y_gt

  # TODO
  def build(self):
    # TODO: build model
    feature_vec = snippet.MRE(self.x,
                              k=ctx.MRE_k,
                              nn_sizes=ctx.MRE_nn_sizes,
                              agg_num=ctx.MRE_agg_num)
    print('[DEBUG] MRE get feature vec: {}'.format(feature_vec.shape))

    #self.optimizer = tf.train.AdamOptimizer(learning_rate=ctx.learning_rate,
    #                                        beta1=ctx.beta1,
    #                                        beta2=ctx.beta2,
    #                                        epsilon=ctx.epsilon)

    #self.train_op = self.optimizer.minimize(self.loss)

