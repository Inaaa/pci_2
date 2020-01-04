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
    #parser.add_argument('--kitti-road-velodyne', type=str, help='Path to KITTI `data_road_velodyne`', default='data_road_velodyne/')
    parser.add_argument('--cam-idx', type=int, help='Index of the camera being used', default=2)
    parser.add_argument('--dataset', type=str, choices=('training', 'testing'), help='Which dataset to run on', default='training')

    args = parser.parse_args()
    dirpath = os.path.join(args.kitti_road, args.dataset, 'velodyne')
    newpath = os.path.join('/home/chli/cc_code/pci/newdata/')
    newpath2 = os.path.join('/home/chli/cc_code/pci/newdata2/')
    print('kitti_road {}'.format(args.kitti_road))
    #print('kitti_road_velodyne {}'.format(args.kitti_road_velodyne))
    print('dirpath {}'.format(dirpath))
    #print(os.listdir(dirpath))
    f = open('/home/chli/cc_code/pci/train.txt', 'a')
    dim =(384,1244)

    for filename in os.listdir(dirpath):
        if filename.startswith('.'):
            continue
        name = filename.split('.')[0]
        print(name)
        f.write(name+'\n')
        id = name.split('_')[1]
        velo_path = os.path.join(args.kitti_road, args.dataset, 'velodyne', filename)
        calib_path = os.path.join(args.kitti_road, args.dataset, 'calib', '%s.txt' % name)
        gt_path = os.path.join(args.kitti_road, args.dataset, 'gt_image_2', 'um_lane_%s.png' % id)
        image_path = os.path.join(args.kitti_road, args.dataset, 'image_2', 'um_%s.png' % id)
        print('calib_path {}'.format(calib_path))
        print(image_path)




        # n x 4 (x, y, z, intensity)
        velo_data = np.fromfile(velo_path, dtype=np.float32).reshape((-1, 4))
        print('velo_data {}'.format(velo_data.shape))
        velo_points = velo_data[:, :3]

        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        resized_gt = cv2.resize(gt, dim, cv2.INTER_NEAREST)
        if gt is None:
            print('\r%s does not have a ground truth file' % filename)
            continue
        gt_labels = gt[:, :, 0]
        print(np.unique(gt_labels))
        h, w = gt_labels.shape
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        resized_image = cv2.resize(image, dim, cv2.INTER_NEAREST)
        print('image {}'.format(image.shape))
        print(gt.shape)
        print(gt_labels.shape)

        calib = Calib(calib_path)
        #aa= calib.get_imu2velo()
        #print('aa {}'.format(aa))

        img_points = calib.velo2img(velo_points, 2).astype(int)
        print(img_points.shape)
        y, x = img_points.T
        print('y = {}'.format(y))
        print('y_shape {}'.format(y.shape))
        print('x = {}'.format(x))
        print('x_shape {}'.format(x.shape))
        selector = (y < h) * (y > 0) * (x < w) * (x > 0)
        print('selector {}'.format(selector.shape))
        filtered_img_points = img_points[selector]
        print('filtered_img_point_shape {}'.format(filtered_img_points.shape))
        velo_new_data = velo_data[selector]

        y, x = np.round(filtered_img_points).astype(int).T
        velo_labels = gt_labels[y, x].reshape(y.shape[0], 1)
        bgr = image[y, x].reshape(y.shape[0], 3)
        velo_data2 = np.hstack((velo_new_data, velo_labels))


        ### generate depth_image
        s =(h, w, 5)
        depth_image = np.zeros(s)
        num = 0
        for i, j in filtered_img_points:
            depth_image[i,j] = velo_data2[num]
            num += 1
        depth_image_array= np.reshape(depth_image, (-1, 5))

        image_array = image.reshape(-1, 3)



        print('depth_image_shape {}'.format(np.unique(depth_image)))
        print('depth_image_shape {}'.format(depth_image_array.shape))
        print('image_array.shape {}'.format(image_array.shape))

        print('velo_lables {}'.format(velo_labels))
        print('bgr  {}'.format(bgr))
        rgb_depth_image = np.hstack((image_array, depth_image_array))
        velo_new_path = os.path.join(newpath2, args.dataset, 'gt_velodyne', name)
        os.makedirs(os.path.dirname(velo_new_path), exist_ok=True)
        np.save(velo_new_path, rgb_depth_image)
        sys.stdout.write("\rConverted %s" % filename)
        sys.stdout.flush()
        break


    print('Done, yay.')
    f.close()









if __name__ == '__main__':
    main()