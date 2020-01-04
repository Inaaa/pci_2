"""project point clouds to  image ."""
from thirdparty.calib import Calib
import argparse
import os
import cv2
import numpy as np
import sys


def main():
    parser = argparse.ArgumentParser(description='Converter')
    parser.add_argument('--kitti-road', type=str, help='Path to KITTI `data_road`', default= \
        '/mrtstorage/datasets/kitti/road_detection/data_road/')
    # parser.add_argument('--kitti-road-velodyne', type=str, help='Path to KITTI `data_road_velodyne`', default='data_road_velodyne/')
    parser.add_argument('--cam-idx', type=int, help='Index of the camera being used', default=2)
    parser.add_argument('--dataset', type=str, choices=('training', 'testing'), help='Which dataset to run on',
                        default='training')

    args = parser.parse_args()
    dirpath = os.path.join(args.kitti_road, args.dataset, 'velodyne')

    newpath3 = os.path.join('/mrtstorage/users/chli/dataset')
    f = open('/home/chli/cc_code/pci/train.txt', 'a')
    dim = (1248, 384)
    aa=0

    for filename in os.listdir(dirpath):
        if filename.startswith('.'):
            continue
        name = filename.split('.')[0]
        #print(name)
        f.write(name + '\n')
        id = name.split('_')[1]
        velo_path = os.path.join(args.kitti_road, args.dataset, 'velodyne', filename)
        calib_path = os.path.join(args.kitti_road, args.dataset, 'calib', '%s.txt' % name)
        gt_path = os.path.join(args.kitti_road, args.dataset, 'gt_image_2', 'um_lane_%s.png' % id)
        image_path = os.path.join(args.kitti_road, args.dataset, 'image_2', 'um_%s.png' % id)
        #print('calib_path {}'.format(calib_path))
        #print(image_path)

        # n x 4 (x, y, z, intensity)
        velo_data = np.fromfile(velo_path, dtype=np.float32).reshape((-1, 4))
        #print('velo_data {}'.format(velo_data.shape))
        velo_points = velo_data[:, :3]

        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        #print('gt_shape{}'.format(gt.shape))

        if gt is None:
            print('\r%s does not have a ground truth file' % filename)
            continue
        gt_labels = gt[:, :, 0]
        #print(gt_labels[0, 0])

        #cv2.imshow('label_image', gt_labels)
        #cv2.waitKey(0)
        #break

        resized_gt_labels = cv2.resize(gt_labels, dim, cv2.INTER_NEAREST)


        #print('resized_gt_labels_shape{}'.format(resized_gt_labels.shape))
        resized_gt =resized_gt_labels.reshape(-1, 1)

        h, w = gt_labels.shape
        #print(h,w)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        resized_image = cv2.resize(image, dim, cv2.INTER_NEAREST)


        calib = Calib(calib_path)

        img_points = calib.velo2img(velo_points, 2).astype(int)
        #print(img_points.shape)
        y, x = img_points.T

        selector = (y < h) * (y > 0) * (x < w) * (x > 0)

        filtered_img_points = img_points[selector]
        velo_new_data = velo_data[selector]
        #print('velo_new_data_shape{}'.format(velo_new_data.shape))
        y, x = np.round(filtered_img_points).astype(int).T
        velo_labels = gt_labels[y, x].reshape(y.shape[0], 1)

        ### generate depth_image
        s = (h, w, 4)
        depth_image = np.zeros(s)
        num = 0
        for i, j in filtered_img_points:
            depth_image[i, j] = velo_new_data[num]
            #print(depth_image)
            num += 1


        resized_depth_image = cv2.resize(depth_image, dim, cv2.INTER_NEAREST)
        resized_depth_image_array=resized_depth_image.reshape(-1,4)
        resized_image_array = resized_image.reshape(-1, 3)

        #print('resized_depth_image {}'.format(np.unique(depth_image)))
        print('resized_depth_image_shape {}'.format(resized_depth_image_array.shape))
        #print('resized_gt {}'.format(resized_gt.shape))
        #print('resized_image_array_shape {}'.format(resized_image_array.shape))


        rgb_depth_image = np.hstack((resized_image_array, resized_depth_image_array, resized_gt))
        print('rgb_depth_image_shape1 {}'.format(rgb_depth_image.shape))
        rgb_depth_image =np.reshape(rgb_depth_image, (384,1248,8))
        print('rgb_depth_image_shape2 {}'.format(rgb_depth_image.shape))
        velo_new_path = os.path.join(newpath3, args.dataset, 'gt_velodyne', str(aa))
        aa=aa+1
        os.makedirs(os.path.dirname(velo_new_path), exist_ok=True)
        np.save(velo_new_path, rgb_depth_image)
        sys.stdout.write("\rConverted %s" % filename)
        sys.stdout.flush()


    print('Done, yay.')
    f.close()


if __name__ == '__main__':
    main()
