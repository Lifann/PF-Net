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

  cloud = cloud.down_sample(12288)
  print('[DEBUG] after down sample, cloud shape: ', cloud.data.shape)
  save_path = 'tmp/{}_sample12288.png'.format(cat)
  vision.show_3d(save_path, cloud)

  num_cropped = int(12288 / 4)
  cloud, cropped_cloud = cloud.crop(num_cropped,
                                    remove_cropped=True,
                                    return_hollowed=True,
                                    reuse=True)
  print('[DEBUG] after crop {}, cloud shape: {}, cropped_cloud: {}'.format(
        num_cropped, cloud.data.shape, cropped_cloud.data.shape))
  save_path = 'tmp/{}_crop{}.png'.format(cat, num_cropped)
  vision.show_3d(save_path, cropped_cloud)
  save_path = 'tmp/{}_incomplete12288.png'.format(cat)
  vision.show_3d(save_path, cloud)
