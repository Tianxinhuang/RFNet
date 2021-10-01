# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import argparse
import csv
import numpy as np
import os
import tensorflow as tf
import time
from io_util import read_pcd, save_pcd
from vv_recon import chamfer_big, earth_mover,full_process,fidelity_loss
from visu_util import plot_pcd_three_views
from data_util import resample_pcd
import time
os.environ["CUDA_VISIBLE_DEVICES"]="0"
def count():
    a=tf.trainable_variables()
    print(a[0].name)
    print(sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables() if not v.name.startswith('E')]))
def test(args):
    inputs = tf.placeholder(tf.float32, (1, 3000, 3))
    npts = tf.placeholder(tf.int32, (1,))
    gt = tf.placeholder(tf.float32, (1, args.num_gt_points, 3))
    _,_,_,complete_op=full_process(inputs)
    count()
    output = tf.placeholder(tf.float32, (1, args.num_gt_points, 3))

    cd_op,_ = chamfer_big(output, gt)
    emd_op = fidelity_loss(inputs, output)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.004#0.003
    sess = tf.Session(config=config)

    var=[v for v in tf.global_variables()]
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=var)
    saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))

    os.makedirs(args.results_dir, exist_ok=True)
    csv_file = open(os.path.join(args.results_dir, 'results.csv'), 'w')
    writer = csv.writer(csv_file)
    writer.writerow(['id', 'cd', 'emd'])

    with open(args.list_path) as file:
        model_list = file.read().splitlines()
    total_time = 0
    total_cd = 0
    total_emd = 0
    cd_per_cat = {}
    emd_per_cat = {}
    for i, model_id in enumerate(model_list):
        partial = read_pcd(os.path.join(args.data_dir, 'partial', '%s.pcd' % model_id))
        complete = read_pcd(os.path.join(args.data_dir, 'complete', '%s.pcd' % model_id))

        partial=resample_pcd(partial,3000)
        start = time.time()
        completion = sess.run(complete_op, feed_dict={inputs: [partial]})
        #print(np.shape(completion))
        mytime=time.time() - start
        print('time',mytime)
        if i>=10:
            total_time += mytime
        cd, emd = sess.run([cd_op, emd_op], feed_dict={inputs: [partial],output: completion, gt: [complete]})
        total_cd += cd
        total_emd += emd
        writer.writerow([model_id, cd, emd])

        synset_id, model_id = model_id.split('/')
        if not cd_per_cat.get(synset_id):
            cd_per_cat[synset_id] = []
        if not emd_per_cat.get(synset_id):
            emd_per_cat[synset_id] = []
        cd_per_cat[synset_id].append(cd)
        emd_per_cat[synset_id].append(emd)
        
        if i % args.plot_freq == 0:
            os.makedirs(os.path.join(args.results_dir, 'plots', synset_id), exist_ok=True)
            plot_path = os.path.join(args.results_dir, 'plots', synset_id, '%s.png' % model_id)
            plot_pcd_three_views(plot_path, [partial, completion[0], complete],
                                             ['input', 'output', 'ground truth'],
                                 'CD %.4f  EMD %.4f' % (cd, emd),
                                 [5, 0.5, 0.5])
            #assert False
        if args.save_pcd:
            os.makedirs(os.path.join(args.results_dir, 'pcds', synset_id), exist_ok=True)
            save_pcd(os.path.join(args.results_dir, 'pcds', '%s.pcd' % model_id), completion[0])
    csv_file.close()
    sess.close()

    print('Average time: %f' % (total_time / (len(model_list)-10)))
    print('Average Chamfer distance: %f' % (total_cd / len(model_list)))
    print('Average Earth mover distance: %f' % (total_emd / len(model_list)))
    print('Chamfer distance per category')
    for synset_id in cd_per_cat.keys():
        print(synset_id, '%f' % np.mean(cd_per_cat[synset_id]))
    print('Earth mover distance per category')
    for synset_id in emd_per_cat.keys():
        print(synset_id, '%f' % np.mean(emd_per_cat[synset_id]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_path', default='../../dense_data/test.list')
    parser.add_argument('--data_dir', default='../../dense_data/test')
    parser.add_argument('--checkpoint', default='./bestrecord')
    parser.add_argument('--results_dir', default='results/recon')
    parser.add_argument('--num_gt_points', type=int, default=16384)
    parser.add_argument('--plot_freq', type=int, default=100)
    parser.add_argument('--save_pcd', action='store_true')
    args = parser.parse_args()

    test(args)

