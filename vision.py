import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm


def show_3d(save_path, cloud, view=None):
  fig = plt.figure(figsize=(13, 13))
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
  if view:
    plt.set_xlim(view[0])
    plt.set_ylim(view[1])
    plt.set_zlim(view[2])
  plt.savefig(save_path)
