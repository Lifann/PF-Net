import vision
import data_loader as dl
import point_cloud as cloud


def test_save_fig(fname):
  arr, color = dl.arrays_from_file(fname)
  cloud.PointCloud(data=arr, color=color)
  vision.show_3d(fname, cloud)


fname = 'test_data/1.txt'
test_save_fig(fname)