import numpy as np

from matplotlib import pyplot as plt


def scatter_points(fname, xyz, color):
  fig = plt.figure(figsize=(100, 100))
  ax = plt.axes(projection='3d')

  ax.scatter3D(xyz[0, :], xyz[1, :], xyz[2, :], color='green')
  plt.savefig(fname)