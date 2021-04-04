import vision
import data_loader as dl
import point_cloud as pcloud


def test_save_fig(src, dest):
  arr, color = dl.arrays_from_file(src)
  cloud = pcloud.PointCloud(data=arr, color=color)
  vision.show_3d(dest, cloud)


src = 'test_data/11.txt'
dest = 'tmp/11.png'
test_save_fig(src, dest)
