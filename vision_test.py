import vision
import data_loader as dl

def test_save_fig(fname):
  arr, color = dl.arrays_from_file(fname)
  vision.scatter_points(fname, arr, color)


fname = 'test_data/1.txt'
test_save_fig(fname)