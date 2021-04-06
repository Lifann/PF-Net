"""
Global context
"""

class Context(object):

  def __init__(self):

    # sys
    self.device = '/CPU:0'

    # General
    self.is_training = True
    self.learning_rate = 0.00001
    self.beta1 = 0.9
    self.beta2 = 0.999
    self.loss_coef = 0.8
    self.epsilon = 1e-05

    # Inputs
    self.num_sampled = 16384
    self.num_incomplete = 12288
    self.num_cropped = 4096

    # MRE
    self.MRE_k = 2
    self.MRE_nn_sizes = [64, 128, 256, 512, 1024]
    self.MRE_agg_num = 4

    # PPD
    self.PPD_M = self.num_cropped
    self.PPD_M1 = 64
    self.PPD_M2 = 128
    self.PPD_FC_sizes = [1024, 512, 256]

    # AD
    self.AD_agg_num = 3

    # GLOSS
    self.GLOSS_coef = 0.2

    # ADLOSS
    self.ADLOSS_CMLP_nn_size = [64, 64, 128, 256]
    self.ADLOSS_agg_num = 3
    self.ADLOSS_nn_size=[256, 128, 16, 1]

    # details
    self.max_pool_kernel = (1, 1, 3, 1)
    self.max_pool_stride = (1, 1, 1, 1)

ctx = Context()
