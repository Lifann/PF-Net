import data_loader as dl


def test_load_arrays_from_file(filename):
  arrs, colors = dl.arrays_from_file(filename)
  print(arrs.shape)

filename = 'test_data/1.txt'
test_load_arrays_from_file(filename)
