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

    #self.optimizer = tf.train.AdamOptimizer(learning_rate=ctx.learning_rate,
    #                                        beta1=ctx.beta1,
    #                                        beta2=ctx.beta2,
    #                                        epsilon=ctx.epsilon)

    #self.train_op = self.optimizer.minimize(self.loss)

