"""
Data structure of on handle of point cloud clutering data.
"""

from copy import deepcopy

import data_loader as dl
import numpy as np
import tensorflow as tf
import traceback
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

  def crop(self, num_cropped, remove_cropped=False, return_hollowed=False, reuse=True):
    """
    Crop part of point cloud.

    Args:
      num_reserved: Points number reserved.
      remove_cropped: bool. If true, remove croped area from data, else set them
        to primary point.
      return_hollowed: bool. Indicate whether if return hollowed part.
      reuse: bool. If true, reuse data space of parent, otherwise create a new
        data space from parent.
    
    Returns:
      PointClouds: remained part and hollowed part (optional).
    """
    if num_cropped > self.length:
      raise ValueError('Reserved points number after crop must be less'
                       ' than cloud points number.')

    viewpoint = utils.random_view()
    distance = utils.distance_to_point(self._data, viewpoint)
    cropped_indices = np.argsort(distance)

    if not reuse:
      data = deepcopy(self._data)
      color = deepcopy(self._color)
    else:
      data = self._data
      color = self._color

    if return_hollowed:
      cropped_data = deepcopy(self._data)
      cropped_color = deepcopy(self._color)

    if not remove_cropped:
      if return_hollowed:
        cropped_data = cropped_data[cropped_indices[:num_cropped]]
        cropped_color = cropped_color[cropped_indices[:num_cropped]]
      data[cropped_indices[:num_cropped], :] = (0., 0., 0.)
    else:
      if return_hollowed:
        cropped_data = cropped_data[cropped_indices[:num_cropped]]
        cropped_color = cropped_color[cropped_indices[:num_cropped]]
      data = np.delete(data, cropped_indices[:num_cropped], axis=0)
      color = np.delete(color, cropped_indices[:num_cropped], axis=0)

    if return_hollowed:
      return (PointCloud(self.category, data, color),
              PointCloud(self.category, cropped_data, cropped_color))
    else:
      return PointCloud(self.category, data, color)

  def tf_crop(self, num_cropped, remove_cropped=False, return_hollowed=False, reuse=True):
    """
    Crop in Tensorflow scope.

    Args:
      num_reserved: Points number reserved.
      remove_cropped: bool. If true, remove croped area from data, else set them
        to primary point.
      return_hollowed: bool. Indicate whether if return hollowed part.
      reuse: bool. If true, reuse data space of parent, otherwise create a new
        data space from parent.
    
    Returns:
      Tuple of tensors. If return_hollowed is true, then return:
        (incomplete data tensor, incomplete color tensor, cropped data tensor, cropped color tensor)
      otherwise:
        (incomplete data tensor, incomplete color tensor)
    """
    viewpoint = utils.random_view()
    distance = utils.tf_distance_to_point(self._data, viewpoint)
    num_reserved = self.length - num_cropped
    _, indices = topk(distance, k=num_reserved, sorted=True)
    _, cropped_indices = topk(distance, k=num_cropped, sorted=False)

    if not reuse:
      raise ValueError('Always reuse in Tensorflow scope.')
    if return_hollowed:
      cropped_data = tf.gather(self._tf_data, cropped_indices)
      cropped_color = tf.gather(self._tf_color, cropped_indices)

    if not remove_cropped:
      raise ValueError('Always remove cropped part in Tensorflow scope,'
                       ' since there is no direct result of missing parts'
                       ' in real prediction job.')
    if return_hollowed:
      return (data, color, cropped_data, cropped_color)
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

  # TODO: Use strategy in: https://arxiv.org/abs/1706.02413.
  # Currently use random strategy.
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

