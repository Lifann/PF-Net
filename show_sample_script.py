import point_cloud as pcloud
import vision

# category : 9 - 12
category = list(range(9, 12))

for cat in category:
  src_filename = 'test_data/{}.txt'.format(cat)
  cloud = pcloud.PointCloud(cat)
  cloud.from_file(src_filename)
  cloud.normalize()

  # show origin figure
  save_path = 'tmp/{}.png'.format(cat)
  vision.show_3d(save_path, cloud)

  cloud = cloud.down_sample(16384)
  save_path = 'tmp/{}_sample{}.png'.format(cat, 16384)
  vision.show_3d(save_path, cloud)

  num_cropped = 16384 - 12288
  cloud, cropped_cloud = cloud.crop(num_cropped,
                                    remove_cropped=True,
                                    return_hollowed=True,
                                    reuse=True)
  save_path = 'tmp/{}_crop{}.png'.format(cat, num_cropped)
  vision.show_3d(save_path, cropped_cloud)
  save_path = 'tmp/{}_incomplete{}.png'.format(cat, 16384)
  vision.show_3d(save_path, cloud)
