"""
Entry point of training program
"""
from context import ctx
from model import Model
from point_cloud import PointCloud

import argparse
import common
import data_loader as dl
import os
import sys
import tensorflow as tf
import time
import utils
import vision


parser = argparse.ArgumentParser()

# train conf
parser.add_argument('--dataset', type=str, default='test_data', help='dataset path')
parser.add_argument('--gpu', type=int, default=-1, help='GPU id used. -1 means not use any GPU.')
parser.add_argument('--epoch_num', type=int, default=200, help='epoch number')
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
  train_files = [os.path.join(train_path, f) for f in os.listdir(train_path)]
  test_files = [os.path.join(test_path, f) for f in os.listdir(test_path)]

  # Build model
  model = build_model()

  '''
  # Load training data
  train_clouds_x = []
  train_clouds_y_bold = []
  train_clouds_y_mid = []
  train_clouds_y_fine = []
  train_clouds = []

  test_clouds_x = []
  test_clouds_y_bold = []
  test_clouds_y_mid = []
  test_clouds_y_fine = []

  # Load training data.
  for idx, afile in enumerate(train_files):
    data, color = dl.arrays_from_file(afile)
    category = utils.index_from_file(afile)
    cloud = PointCloud(category, data=data, color=color)
    cloud.normalize()

    cloud = cloud.down_sample(ctx.num_sampled)
    train_clouds.append(cloud)

    incomplete_cloud, cropped_cloud = cloud.crop(
        ctx.num_cropped,
        remove_cropped=True,
        return_hollowed=True,
        reuse=True)
    cropped_cloud_bold, cropped_cloud_mid, cropped_cloud_fine =  \
        common.get_multi_resolution_clouds(cropped_cloud)
    print('[info] crop data: incomplete part: {}, cropped part: {}'.format(
          incomplete_cloud.data.shape, cropped_cloud.data.shape))

    train_clouds_x.append(incomplete_cloud)
    train_clouds_y_bold.append(cropped_cloud_bold)
    train_clouds_y_mid.append(cropped_cloud_mid)
    train_clouds_y_fine.append(cropped_cloud_fine)
    print('[info] Load train data: {}'.format(afile))

  ## Load testing data for check.
  for idx, afile in enumerate(test_files):
    data, color = dl.arrays_from_file(afile)
    category = utils.index_from_file(afile)
    cloud = PointCloud(category, data=data, color=color)
    cloud.normalize()

    cloud = cloud.down_sample(ctx.num_sampled)

    incomplete_cloud, cropped_cloud = cloud.crop(
        ctx.num_cropped,
        remove_cropped=True,
        return_hollowed=True,
        reuse=True)
    cropped_cloud_bold, cropped_cloud_mid, cropped_cloud_fine =  \
        common.get_multi_resolution_clouds(cropped_cloud)

    test_clouds_x.append(incomplete_cloud)
    test_clouds_y_bold.append(cropped_cloud_bold)
    test_clouds_y_mid.append(cropped_cloud_mid)
    test_clouds_y_fine.append(cropped_cloud_fine)
    print('[info] Load test data: {}'.format(afile))

  config = tf.ConfigProto()
  config.inter_op_parallelism_threads = 0
  config.intra_op_parallelism_threads = 0
  session = tf.Session(config=config)
  session.run(tf.global_variables_initializer())

  #### show at start
  index = train_files.index('test_data/train/9.txt')
  mark = 9
  print('[debug] index = ', index)
  save_path = 'tmp/data_{}_sampled.png'.format(mark)
  vision.show_3d(save_path, train_clouds[index])

  save_path = 'tmp/data_{}_incomplete.png'.format(mark)
  vision.show_3d(save_path, train_clouds_x[index])

  save_path = 'tmp/data_{}_cropped.png'.format(mark)
  vision.show_3d(save_path, train_clouds_y_fine[index])
  ####

  step = 0
  for epoch in range(args.epoch_num):

    for idx, _ in enumerate(train_clouds_x):
      start_time = time.time()
      # TODO: re-design model
      #_, loss = session.run(
      #    [model.train_op, model.loss],
      #    feed_dict={model.x: train_clouds_x[idx].data,
      #               model.y_gt_bold: train_clouds_y_bold[idx].data,
      #               model.y_gt_mid: train_clouds_y_mid[idx].data,
      #               model.y_gt_fine: train_clouds_y_fine[idx].data})
      step += 1
      end_time = time.time()
      step_cost = end_time - start_time
      print('[info] epoch: {}, step={}, loss={}'.format(epoch, step, loss))
      if step % 100 == 0:
        # TODO: re-design model
        #data = session.run(
        #    model.y_fine,
        #    feed_dict={model.x: train_clouds_x[idx].data})
        #save_path = 'tmp/train/cat{}_step{}_pred.png'.format(mark, step)
        #vision.show_3d_data(save_path, data, train_clouds_y_fine[index].color)
  '''
      

if __name__ == '__main__':
  train()
  print('ok')
