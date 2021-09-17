import numpy as np
import argparse
import os
import tensorflow as tf
from io_util import read_pcd, save_pcd
from tf_ops.CD import tf_nndistance
from visu_util import plot_pcd_three_views
from data_util import resample_pcd
import random
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def chamfer_big(pcd1, pcd2):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    return (dist1 + dist2) / 2

def fidelity(pcd1,pcd2):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    result= tf.minimum(dist1,dist2)
    return dist1
#source_data:batch*16384*3
#target)data:16384*3
def MMD(source_data,target_data):
    bnum=tf.shape(source_data)[0]
    data=tf.tile(tf.expand_dims(target_data,axis=0),[bnum,1,1])
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(data, source_data)
    dist1 = tf.reduce_mean(tf.sqrt(dist1),axis=-1)
    dist2 = tf.reduce_mean(tf.sqrt(dist2),axis=-1)
    result=(dist1+dist2)/2
    result=tf.reduce_min(result)
    return result
def get_partial(path, car_id):
    pcd_dir=os.path.join(path,'cars')
    bbox_dir=os.path.join(path,'bboxes')
    partial = read_pcd(os.path.join(pcd_dir, '%s.pcd' % car_id))
    bbox = np.loadtxt(os.path.join(bbox_dir, '%s.txt' % car_id))
    

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
    return partial
def test(args):
    partial=tf.placeholder(tf.float32, (None, 3))
    inputs = tf.placeholder(tf.float32, (16384, 3))
    datapool=tf.placeholder(tf.float32, (args.bnum, 16384, 3))

    tempmmd=MMD(datapool,inputs)
    fidelity_loss=fidelity(tf.expand_dims(partial,axis=0),tf.expand_dims(inputs,axis=0))
    mmdlist=[]
    fidelist=[]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    with open('kitti_record.txt','r') as f:
        showlist=f.readlines()
    showlist=showlist[0].split(',')[:-1]
    idlist=[]
    newlist=[]
    #print('frame_291_car_10' in showlist)
    #assert False

    #with open(args.kittilist) as file0:
        #kitti_list=file0.read().splitlines()
    kitti_list=[filename.split('.')[0] for filename in os.listdir(args.kitti_dir)]
    random.shuffle(kitti_list)
    kittinum=len(kitti_list)

        #with open(args.carlist) as file:
            #car_list = file.read().splitlines()
    car_list=[filename.split('.')[0] for filename in os.listdir(args.car_dir)]
    carnum=len(car_list)
    if args.carnum is not 0:
        car_list = random.shuffle(car_list)
        car_list=car_list[:args.carnum]
        carnum=args.carnum
    itertime=int(carnum/args.bnum)
    #cardata=np.zeros((carnum,16384,3))
    car_list.extend(car_list[:(1+itertime)*args.bnum-carnum])
    carnum=len(car_list)
    cardata=np.zeros((carnum,16384,3))
    for k in range(carnum):
        car_id=car_list[k]
        cardata[k]=read_pcd(os.path.join(args.car_dir, '%s.pcd' % car_id))
        if (car_id in showlist):
            idlist.append(k)
            newlist.append(car_id)
    f=open('kitti_result.txt','w')
    s=open('kitti_show.txt','w')
    for i in range(kittinum):
        kitti_id=kitti_list[i]
        #kittipar=read_pcd(os.path.join(args.kittipar_dir, '%s.pcd' % kitti_id))
        kittipar=get_partial(args.kittipar_dir,kitti_id)
        kittidata=read_pcd(os.path.join(args.kitti_dir, '%s.pcd' % kitti_id))
        #kittipar=resample_pcd(kittipar,2048)
        fide=sess.run(fidelity_loss,feed_dict={inputs:kittidata,partial:kittipar})
        #print('fidelity: ',fide)
        f.write('fidelity: '+str(fide)+'\n')
        fidelist.append(fide)
        templist=[]
        for j in range(carnum//args.bnum):
            #cardata=np.zeros((args.bnum,16384,3))
            #for k in range(args.bnum):
            #    car_id=car_list[j*args.bnum+k]
            #    cardata[k]=read_pcd(os.path.join(args.car_dir, '%s.pcd' % car_id))
            temp=sess.run(tempmmd,feed_dict={inputs:kittidata,datapool:cardata[j*args.bnum:(j+1)*args.bnum]})
            templist.append(temp)
            #fide=sess.run(tempmmd,feed_dict={inputs:kittidata,partial:kittipar})
            #fide_list.append(fide)
        mmdlist.append(min(templist))
        if (i in idlist):
            s.write(str(i)+'   '+kitti_id+' fidelity: '+str(fide)+'\n')
            s.write(str(i)+'   '+kitti_id+'MMD: '+str(min(templist))+'\n')
            s.flush()
        #print('MMD: ', min(templist))
        f.write('MMD: '+str(min(templist))+'\n')
        f.flush()
        
    mmd=sum(mmdlist)/len(mmdlist)
    #print('mean MMD: ',mmd)
    f.write('mean MMD: '+str(mmd)+'\n')
    fideloss=sum(fidelist)/len(fidelist)
    f.write('mean Fidelity: '+str(fideloss)+'\n')
    s.close()
    f.close()
    #print('mean Fidelity: ',fideloss)
    sess.close()
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kittipar_dir', default='/home/xk/codetest/kitti')
    parser.add_argument('--kitti_dir', default='kitti_result1')
    parser.add_argument('--kitti_list', default='./kitti_result/kitti.list')
    parser.add_argument('--carlist', default='/home/xk/codetest/kitti/kitti_car/car.list')
    parser.add_argument('--bnum', type=int, default=512)
    parser.add_argument('--car_dir', default='/home/xk/codetest/kitti_cars')
    parser.add_argument('--carnum', type=int, default=0)
    args = parser.parse_args()

    test(args)
