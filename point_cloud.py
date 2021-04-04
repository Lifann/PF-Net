"""
Data structure of on handle of point cloud clutering data.
"""

import data_loader as dl
import numpy as np
import traceback
import tensorflow as tf


class PointCloud(object):
  """
  Point cloud data structure. The scale is always normalized.
  """

  def __init__(self, data=None, color=None):
    """
    Create a point cloud

    Args:
      data: coordinate list marked the point cloud.
      color: color list marked the point cloud color.
    """
    if data is None:
      self._data = None
    else:
      if len(data.shape) != 2:
        raise ValueError('Point cloud shape must be rank 2.')
      if data.shape[1] != 3:
        raise ValueError('Each point in point cloud must dim-3.')
      self._data = np.array(data, dtype=np.float32) # Make a copy.

    if color is None:
      self._color = None
    else:
      if len(color.shape) != 2:
        raise ValueError('Point cloud color shape must be rank 2.')
      if data.shape[1] != 3:
        raise ValueError('Each point in point cloud must dim-3.')
      self._color = np.array(color, dtype=np.float32) # Make a copy.

  def from_file(self, filename):
    """
    Get a point cloud from file.
    """
    self._data, self._color = dl.arrays_from_file(filename)

  # TODO
  def hollow(self, radius):
    """
    Hollow part of point cloud.

    Args:
      radius: Radius of hollowed part.
    
    Returns:
      Tuple of PointClouds: hollowed part, and remained part.
    """
    pass

  # TODO 
  def down_sample(self, num_points, mode='center_random'):
    """
    Get a new down sampled PointCloud from self.
    
    Args:
      num_points: int. Number of down sampled point cloud.
    
    Returns:
      An PointCloud object, down sampled, if it is legal
      to execute down sampling.
    """
    if num_points < self.length:
      return None

  def normalize(self):
    """
    Normalize data
    """
    self._data = tf.keras.utils.normalize(self._data, axis=1)

  @property
  def data(self):
    return self._data

  @property
  def color(self):
    return self._color

  @property
  def length(self):
    return self._data.shape[0]

  # TODO
  def show(self):
    """
    """
    pass

  def save(self, fpath):
    """
    Save self to file.
    """
    try:
      with open(fpath, 'w') as f:
        for i, d in enumerate(self._data):
          x, y, z = d
          if self._color is None:
            R, G, B = 0, 0, 0
          else:
            c = self._color[i]
            R, G, B = c
          # TODO: Add format control
          line = '{}, {}, {}, {}, {}, {}\n'.format(x, y, z, R, G, B)
          f.write(line)
    except:
      print(traceback.format_exc())

