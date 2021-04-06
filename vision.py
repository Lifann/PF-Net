import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from matplotlib import cm


def show_3d(save_path, cloud, view=None):
  fig = plt.figure(figsize=(13, 13))
  ax = plt.axes(projection='3d')
  cloud.normalize()
  color = cloud.color / 255
  xyz = cloud.data

  xl = (np.min(xyz[:, 0]), np.max(xyz[:, 0]))
  yl = (np.min(xyz[:, 1]), np.max(xyz[:, 1]))
  zl = (np.min(xyz[:, 2]), np.max(xyz[:, 2]))

  ax.scatter(xs=xyz[:, 0],
             ys=xyz[:, 1],
             zs=xyz[:, 2],
             s=20,
             alpha=1.0,
             c=color,
             marker='o')

  ax.set_xlim(xl)
  ax.set_ylim(yl)
  ax.set_zlim(zl)
  plt.savefig(save_path)


def _normalize(xyz):
    return tf.keras.utils.normalize(xyz, axis=1)


def show_3d_data(save_path, data, color, view=None):
  fig = plt.figure(figsize=(13, 13))
  ax = plt.axes(projection='3d')
  data = _normalize(data)
  color = color / 255
  xyz = data

  xl = (np.min(xyz[:, 0]), np.max(xyz[:, 0]))
  yl = (np.min(xyz[:, 1]), np.max(xyz[:, 1]))
  zl = (np.min(xyz[:, 2]), np.max(xyz[:, 2]))

  ax.scatter(xs=xyz[:, 0],
             ys=xyz[:, 1],
             zs=xyz[:, 2],
             s=20,
             alpha=1.0,
             c=color,
             marker='o')

  ax.set_xlim(xl)
  ax.set_ylim(yl)
  ax.set_zlim(zl)
  plt.savefig(save_path)
  plt.close()
