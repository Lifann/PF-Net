import point_cloud as pcloud
import vision

# category : 1 - 15
category = list(range(1, 16))

for cat in category:
  src_filename = 'test_data/{}.txt'.format(cat)
  cloud = PointCloud(category)
  cloud.from_file(src_filename)
  cloud.normalize()

  save_path = 'tmp/{}.png'.format(cat)
  vision.show_3d(save_path, cloud)
