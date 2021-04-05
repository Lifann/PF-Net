"""
Data structure of on handle of point cloud clutering data.
"""

import data_loader as dl
import numpy as np
import traceback
import tensorflow as tf
import utils


class PointCloud(object):
  """
  Point cloud data structure. The scale is always normalized.
  """

  def __init__(self, category, data=None, color=None):
    """
    Create a point cloud

    Args:
      data: coordinate list marked the point cloud.
      color: color list marked the point cloud color.
    """
    if not isinstance(category, int):
      raise TypeError('category should be an integer which indicates'
                      'point cloud mass.')
    self.category = category

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

      self._tf_data = tf.placeholder(tf.float32)
      self._tf_color = tf.placeholder(tf.int32)

  def from_file(self, filename):
    """
    Get a point cloud from file.
    """
    self._data, self._color = dl.arrays_from_file(filename)

  # TODO
  def crop(self, num_reserved, return_hollowed=False):
    """
    Crop part of point cloud.

    Args:
      num_reserved: Points number reserved.
      return_hollowed: bool. Indicate whether if return hollowed part.
    
    Returns:
      PointClouds: remained part and hollowed part (optional).
    """
    if num_reserved > self.length:
      raise ValueError('Reserved points number after crop must be less'
                       ' than cloud points number.')

    viewpoint = utils.random_view()
    distance = distance_to_point(self._data, viewpoint)
    indices = np.argsort(-distance)

    data = self._data[indices[:num_reserved]]
    color = self._color[indices[:num_reserved]]
    return PointCloud(self.category, data, color)

  def tf_crop(self, num_reserved, return_hollowed=False):
    """
    Crop in Tensorflow scope. 

    Args:
      numreserved
    
    Returns:
      Tuple of tensors: croped (data, color)
    """
    viewpoint = utils.random_view()
    distance = distance_to_point(self._data, viewpoint)
    _, indices = topk(distance k=num_reserved, sorted=True)

    data = tf.gather(self._tf_data, indices)
    color = tf.gather(self._tf_color, indices)
    return (data, color)

  def down_sample(self, num_points):
    """
    Get a new down sampled PointCloud from self, uniformly.
    
    Args:
      num_points: int. Number of down sampled point cloud.
    
    Returns:
      An PointCloud object, down sampled, if it is legal
      to execute down sampling.
    """
    if num_points > self.length:
      return None
    else:
      indices = np.random.choice(self.length, num_points, replace=False)
      return PointCloud(self.category, self.data[indices], self.color[indices])

  def tf_down_sample(self, num_points, return_color=False):
    """
    Down sampling in Tensorflow scope for better performance.

    Returns:
      Downsampled data and color tensors.
    """
    if num_points > self.length:
      return None
    else:
      indices = tf.random.uniform((num_points, ), minval=0, maxval=self.length, dtype=tf.int32)
      down_sampled_data = tf.gather(self.tf_data, indices)
      down_sampled_color = tf.gather(self.tf_color, indices)

      if return_color:
        return (down_sampled_data, down_sampled_color)
      return down_sampled_data

  def normalize(self):
    """
    Normalize data
    """
    self._data = tf.keras.utils.normalize(self._data, axis=1)

  @property
  def data(self):
    return self._data

  @property
  def tf_data(self):
    return self._tf_data

  @property
  def color(self):
    return self._color

  @property
  def tf_color(self):
    return self._tf_color

  @property
  def length(self):
    return self._data.shape[0]

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

