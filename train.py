"""
Entry point of training program
"""
from context import ctx
from model import Model
from point_cloud import PointCloud

import argparse
import data_loader as dl
import os
import tensorflow as tf
import utils


parser = argparse.ArgumentParser()

# train conf
parser.add_argument('--dataset', type=str, default='test_data', help='dataset path')
parser.add_argument('--gpu', type=int, default=-1, help='GPU id used. -1 means not use any GPU.')
parser.add_argument('--epoch_num', type=int, default=1, help='epoch number')
parser.add_argument('--check_interval', type=int, default=10, help='steps interval for check')

# hyper-params
parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning_rate')
parser.add_argument('--weight_decay', type=float, default=0.001)
args = parser.parse_args()

ctx.device = '/CPU:0' if args.gpu < 0 else '/GPU:{}'.format(args.gpu)


def build_model():
  model = Model()
  model.build()
  return model


# TODO
def train():
  data_root = args.dataset
  train_path = os.path.join(data_root, 'train')
  test_path = os.path.join(data_root, 'test')
  train_files = os.listdir(train_path)
  test_files = os.listdir(test_path)

  # Build model
  model = build_model()

  # Load training data
  train_clouds_x = []
  train_clouds_y_bold = []
  train_clouds_y_mid = []
  train_clouds_y_fine = []

  test_clouds_x = []
  test_clouds_y_bold = []
  test_clouds_y_mid = []
  test_clouds_y_fine = []

  # Load training data.
  for idx, afile in enumerate(train_files):
    data, color = dl.arrays_from_file(afile)
    category = utils.index_from_file(afile)
    cloud = PointCloud(category, data=data, color=color)
    cloud = cloud.normalize()

    cloud = cloud.down_sample(ctx.num_sampled)

    incomplete_cloud, cropped_cloud = cloud.crop(
        ctx.num_cropped,
        remove_cropped=True,
        return_holowed=True,
        reuse=True)
    cropped_cloud_bold, cropped_cloud_mid, cropped_cloud_fine =  \
        common.get_multi_resolution_clouds(cropped_cloud)

    train_clouds_x.append(incomplete_cloud)
    train_clouds_y_bold.append(cropped_cloud_bold)
    train_clouds_y_mid.append(cropped_cloud_mid)
    train_clouds_y_fine.append(cropped_cloud_fine)

  ## Load testing data for check.
  for idx, afile in enumerate(train_files):
    data, color = dl.arrays_from_file(afile)
    category = utils.index_from_file(afile)
    cloud = PointCloud(category, data=data, color=color)
    cloud = cloud.down_sample(ctx.num_sampled)
    incomplete_cloud, cropped_cloud = cloud.crop(
        ctx.num_cropped,
        remove_cropped=True,
        return_holowed=True,
        reuse=True)

    train_clouds_x.append(incomplete_cloud)
    train_clouds_y.append(cropped_cloud)

  #session = tf.Session()

  #for epoch in range(args.epoch_num):
  #  for idx, cloud in enumerate(train_clouds):
  #    session.run(model.train_op, feed_dict={model.x: cloud.data,
  #                                           model.y_gt: })

if __name__ == '__main__':
  train()
