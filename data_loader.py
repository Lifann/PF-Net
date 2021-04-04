"""
Data loader is used to load data file into memory
with `numpy.array` format.
"""

import numpy as np
import os
import sys
import traceback


def arrays_from_file(filename): 
  """
  Get numpy arrays from file. Raise ValueError if file is
  set in wrong format.

  Args:
    filename: string. Location of data file.

  Returns:
    Tuple of numpy.array: (point cloud coordinates, colors).
  """
  try:
    arrs, colors = [], []
    with open(filename) as f:
      for line in f:
        if '#' in line:
          continue
        raw_items = line.split()
        x, y, z, R, G, B = [x.strip(',') for x in raw_items]
        arrs.append([float(x), float(y), float(z)])
        colors.append([int(R), int(G), int(B)])
      return (np.array(arrs, dtype=np.float32), np.array(colors, dtype=np.int32))
  except:
    raise ValueError('Failed load array from file. Please check'
                     ' file: {}'.format(filename))

