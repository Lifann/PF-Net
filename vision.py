import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm


def show_3d(save_path, cloud, 3d_space=None):
  fig = plt.figure(figsize=(100, 100))
  ax = plt.axes(projection='3d')
  cloud.normalize()
  color = cloud.color / 255
  xyz = cloud.data

  ax.scatter(xs=xyz[:, 0],
             ys=xyz[:, 1],
             zs=xyz[:, 2],
             s=12,
             alpha=1.0,
             c=color,
             marker='o')
  if 3d_space:
    plt.set_xlim(3d_space[0])
    plt.set_ylim(3d_space[1])
    plt.set_zlim(3d_space[2])
  plt.savefig(save_path)
