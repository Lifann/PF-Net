import numpy as np
import random


def random_view():
  choice = [
      np.array((1, 0, 0), dtype=np.float32),
      np.array((0, 0, 1), dtype=np.float32),
      np.array((1, 0, 1), dtype=np.float32),
      np.array((-1, 0, 0), dtype=np.float32),
      np.array((-1, 1, 0), dtype=np.float32),
  ]
  # Randomly choose a viewpoint.
  return random.sample(choice, 1)


def distance_to_point(arr, point):
  tmp = arr - point
  return np.sum(tmp * tmp, axis=1)


def tf_distance_to_point(arr, point):
  tmp = arr - point
  return tf.reduce_sum(tmp * tmp, axis=1)


def size_from_shape(shape):
  t = 1
  for x in shape:
    t *= x
  return t
