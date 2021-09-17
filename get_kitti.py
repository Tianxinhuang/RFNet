

import argparse
import importlib
#import models
import numpy as np
import os
import tensorflow as tf
import time
from io_util import read_pcd, save_pcd
from visu_util import plot_pcd_three_views
from vv_recon import full_process
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def test(args):
    inputs = tf.placeholder(tf.float32, (1, None, 3))
    npts = tf.placeholder(tf.int32, (1,))
    gt = tf.placeholder(tf.float32, (1, args.num_gt_points, 3))
    _,_,_,outputs=full_process(inputs)
    #model_module = importlib.import_module('.%s' % args.model_type, 'models')
    #model = model_module.Model(inputs, npts, gt, tf.constant(1.0))

    os.makedirs(os.path.join(args.results_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'completions'), exist_ok=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))

    car_ids = [filename.split('.')[0] for filename in os.listdir(args.pcd_dir)]
    total_time = 0
    total_points = 0
    for i, car_id in enumerate(car_ids):
        partial = read_pcd(os.path.join(args.pcd_dir, '%s.pcd' % car_id))
        bbox = np.loadtxt(os.path.join(args.bbox_dir, '%s.txt' % car_id))
        total_points += partial.shape[0]

        # Calculate center, rotation and scale
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale

        partial = np.dot(partial - center, rotation) / scale
        partial = np.dot(partial, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        start = time.time()
        completion = sess.run(outputs, feed_dict={inputs: [partial]})
        total_time += time.time() - start
        completion = completion[0]

        completion_w = np.dot(completion, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        completion_w = np.dot(completion_w * scale, rotation.T) + center
        pcd_path = os.path.join(args.results_dir, '%s.pcd' % car_id)
        save_pcd(pcd_path, completion)

        if i % args.plot_freq == 0:
            plot_path = os.path.join(args.results_dir, 'plots', '%s.png' % car_id)
            plot_pcd_three_views(plot_path, [partial, completion], ['input', 'output'],
                                 '%d input points' % partial.shape[0], [5, 0.5])
    print('Average # input points:', total_points / len(car_ids))
    print('Average time:', total_time / len(car_ids))
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='RFNet')
    parser.add_argument('--checkpoint', default='./modelvv_recon')
    parser.add_argument('--pcd_dir', default='/home/xk/codetest/kitti/cars')
    parser.add_argument('--bbox_dir', default='/home/xk/codetest/kitti/bboxes')
    parser.add_argument('--results_dir', default='kitti_result1')
    parser.add_argument('--num_gt_points', type=int, default=16384)
    parser.add_argument('--plot_freq', type=int, default=100)
    parser.add_argument('--save_pcd', action='store_true')
    args = parser.parse_args()

    test(args)
