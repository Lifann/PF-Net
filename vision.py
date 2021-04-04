import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm


def show_3d(save_path, cloud):
  fig = plt.figure(figsize=(100, 100))
  ax = plt.axes(projection='3d')
  cloud.normalize()
  color = cloud.color / 255
  xyz = cloud.data

  ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color)
  plt.savefig(save_path)
