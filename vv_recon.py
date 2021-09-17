# -*- coding: utf-8 -*-

import tensorflow as tf
from numpy import *
import os
import getdata
#import tf_util
import copy
import random
import point_choose

from tf_ops.emd import tf_auctionmatch
from tf_ops.CD import tf_nndistance
from tf_ops.sampling import tf_sampling
from tf_ops.grouping import tf_grouping

from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
from tf_ops.grouping.tf_grouping import query_ball_point, group_point


from tf_ops.interpolation.tf_interpolate import three_nn,three_interpolate 
from data_util import lmdb_dataflow
from visu_util import plot_pcd_three_views
from pc_distance import tf_approxmatch
#from test_emd import iter_match, val_match

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
#sys.path.append(os.path.join(BASE_DIR, './DP_PCN'))
#import transform_nets as tn
#from transform_nets import my_transform_net
#from encoders_decoders import batch_normalization,dropout
#from provider import shuffle_data,shuffle_points,rotate_point_cloud,jitter_point_cloud
#DATA_DIR=getdata.getspdir()
#filelist=os.listdir(DATA_DIR)

#trainfiles=getdata.getfile(os.path.join(DATA_DIR,'train_files.txt'))
##testfiles=getdata.getfile(os.path.join(DATA_DIR,'test_files.txt'))32
#EPOCH_ITER_TIME=101
BATCH_ITER_TIME=300000
#BATCH_ITER_TIME=20
BASE_LEARNING_RATE=0.001
REGULARIZATION_RATE=0.00001
BATCH_SIZE=32
EVAL_SIZE=4
DECAY_STEP=1000*BATCH_SIZE
DECAY_RATE=0.7
INNUM=3000
PTNUM=16384
FILE_NUM=6
os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf.set_random_seed(1)
def get_weight_variable(shape,stddev,name,regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)):
    #print(shape)
    weight = tf.get_variable(name=name,shape=shape,initializer=tf.contrib.layers.xavier_initializer())
    tf.summary.histogram(name+'/weights',weight)
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weight))
    return weight
def get_bias_variable(shape,value,name):
    bias=tf.Variable(tf.constant(value, shape=shape, name=name,dtype=tf.float32))
    tf.summary.histogram(name+'/bias',bias)
    return bias
def get_learning_rate(step):
    learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE, step,DECAY_STEP / BATCH_SIZE, DECAY_RATE, staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)
    return learning_rate
def conv2d(scope,inputs,num_outchannels,kernel_size,stride=[1,1],padding='SAME',stddev=1e-3,use_bnorm=False,activation_func=tf.nn.relu):
    with tf.variable_scope(scope):
        kernel_h,kernel_w=kernel_size
        num_inchannels=inputs.get_shape()[-1].value
        kernel_shape=[kernel_h,kernel_w,num_inchannels,num_outchannels]
        kernel=get_weight_variable(kernel_shape,stddev,'weights')
        stride_h,stride_w=stride
        outputs=tf.nn.conv2d(inputs,kernel,[1,stride_h,stride_w,1],padding=padding)
        bias = get_bias_variable([num_outchannels],0,'bias')
        outputs=tf.nn.bias_add(outputs,bias)
        if use_bnorm:
            outputs=tf.contrib.layers.batch_norm(outputs,
                                      center=True, scale=True,
                                      updates_collections=None,
                                      scope='bn')
        if activation_func!=None:
            outputs=activation_func(outputs)
    return outputs

def fully_connect(scope,inputs,num_outputs,stddev=1e-3,use_bnorm=False,activation_func=tf.nn.relu):
    num_inputs = inputs.get_shape()[-1].value
    # print(inputs,num_inputs)
    with tf.variable_scope(scope):
        weights=get_weight_variable([num_inputs,num_outputs],stddev=stddev,name='weights')
        bias=get_bias_variable([num_outputs],0,'bias')
        result=tf.nn.bias_add(tf.matmul(inputs,weights),bias)
        if use_bnorm:
            outputs=tf.contrib.layers.batch_norm(outputs,
                                      center=True, scale=True,
                                      updates_collections=None,
                                      scope='bn')
        if(activation_func is not None):
            result=activation_func(result)
    return result
def deconv(scope,inputs,output_shape,kernel_size,stride=[1,1],padding='SAME',stddev=1e-3,activation_func=tf.nn.relu):
    with tf.variable_scope(scope) as sc:
        kernel_h,kernel_w=kernel_size
        num_outchannels=output_shape[-1]
        num_inchannels=inputs.get_shape()[-1].value
        kernel_shape=[kernel_h,kernel_w,num_outchannels,num_inchannels]
        kernel=get_weight_variable(kernel_shape,stddev,'weights')
        stride_h,stride_w=stride
        outputs=tf.nn.conv2d_transpose(inputs,filter=kernel,output_shape=output_shape,strides=[1,stride_h,stride_w,1],padding=padding)
        bias = get_bias_variable([num_outchannels],0,'bias')
        outputs = tf.nn.bias_add(outputs, bias)
        if activation_func != None:
            outputs = activation_func(outputs)
    return outputs
def ful_maxpooling(tensor):
    return tf.reduce_max(tensor,axis=1)
def get_voxel(fileNum,voxel_size):
    data=tf_util.load_data(os.path.join(DATA_DIR,'voxel_file'+str(fileNum)+'.npy'))
    return data['voxel'+str(voxel_size)]
def get_covariance(fileNum):
    data = tf_util.load_data(os.path.join(DATA_DIR, 'voxel_file' + str(fileNum) + '.npy'))
    return data['covar_data']
def loss_func(input_voxel,output_voxel):
    model_cross=tf.reduce_sum(input_voxel*tf.log(tf.clip_by_value(output_voxel,1e-10,1.0))+(1.0-input_voxel)*tf.log(tf.clip_by_value(1.0-output_voxel,1e-10,1.0)),axis=-1)
    result=-tf.reduce_mean(model_cross)
    return result
def loss_func_l2(input_voxel,output_voxel):
    return tf.reduce_sum(tf.square(input_voxel-output_voxel))
def sampling(npoint,xyz,use_type='f'):
    if use_type=='f':
        idx=tf_sampling.farthest_point_sample(npoint, xyz)
        new_xyz=tf_sampling.gather_point(xyz,idx)
    elif use_type=='r':
        #bnum=tf.shape(xyz)[0]
        #ptnum=xyz.get_shape()[1].value
        #ptids=arange(ptnum)
        #random.shuffle(ptids)
        #ptid=tf.tile(tf.constant(ptids[:npoint],shape=[1,npoint,1],dtype=tf.int32),[bnum,1,1])
        #bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,npoint,1])
        #idx=tf.concat([bid,ptid],axis=-1)
        #new_xyz=tf.gather_nd(xyz,idx)

        bnum=tf.shape(xyz)[0]
        ptnum=xyz.get_shape()[1].value
        ptids=arange(ptnum)
        ptids=tf.random_shuffle(ptids,seed=None)
        #random.shuffle(ptids)
        #print(ptids,ptnum,npoint)
        #ptidsc=ptids[tf.py_func(np.random.choice(ptnum,npoint,replace=False),tf.int32)]
        ptidsc=ptids[:npoint]
        ptid=tf.cast(tf.tile(tf.reshape(ptidsc,[-1,npoint,1]),[bnum,1,1]),tf.int32)
        #ptid=tf.tile(tf.constant(ptidsc,shape=[1,npoint,1],dtype=tf.int32),[bnum,1,1])

        bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,npoint,1])
        idx=tf.concat([bid,ptid],axis=-1)
        new_xyz=tf.gather_nd(xyz,idx)
    return idx,new_xyz
def attention_sample(npoint,xyz,featvec,global_feat,mlp,use_feat=True):
    featnum=xyz.get_shape()[1].value
    if use_feat and featvec is not None:
        print(xyz,featvec,global_feat)
        tensor=tf.concat([xyz,featvec,tf.tile(tf.expand_dims(global_feat,axis=1),multiples=[1,featnum,1])],axis=-1)
    else:
        tensor=tf.concat([xyz,tf.tile(tf.expand_dims(global_feat,axis=1),multiples=[1,featnum,1])],axis=-1)
    tensor=tf.expand_dims(tensor,axis=2)
    for i,outchannel in enumerate(mlp):
        tensor=conv2d('atten_layer%d'%i,tensor,outchannel,[1,1],padding='VALID')
    eva=conv2d('atten_ac',tensor,1,[1,1],padding='VALID',activation_func=tf.nn.relu)#
    _,idx=tf.nn.top_k(tf.squeeze(eva),npoint)
    batch_index = tf.constant(value=[i for i in range(BATCH_SIZE)], shape=[BATCH_SIZE, 1, 1])
    batch_index = tf.tile(batch_index,multiples=[1,npoint,1])
    index=tf.concat([batch_index,tf.expand_dims(idx,axis=-1)],axis=-1)
    new_xyz=tf.gather_nd(xyz,index)
    return index,new_xyz
def grouping(xyz,new_xyz, mani_xyz, new_mani_xyz, radius, nsample, points, knn=False, use_xyz=True):
    if knn:
        _,idx = tf_grouping.knn_point(nsample, mani_xyz, new_mani_xyz)
    else:
        idx, pts_cnt = tf_grouping.query_ball_point(radius, nsample, mani_xyz, new_mani_xyz)
    grouped_xyz = tf_grouping.group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = tf_grouping.group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return idx,new_points
def local_net(scope,sample_dim,xyz,featvec,r_list,k_list,layers_list,use_xyz=True,use_all=False):
    with tf.variable_scope(scope) as sc:
        newfeatvec=[]
        if use_all:
            ptnum=xyz.get_shape()[1].value
            centers_batch=tf.tile(tf.constant([i for i in range(BATCH_SIZE)],shape=[BATCH_SIZE,1,1]),multiples=[1,ptnum,1])
            centers_pt=tf.tile(tf.constant([i for i in range(ptnum)],shape=[1,ptnum,1]),multiples=[BATCH_SIZE,1,1])
            centers_id=tf.concat([centers_batch,centers_pt],axis=-1)
            centers_coor=xyz 
        else:
            centers_id=point_choose.farthest_sampling(sample_dim,xyz,featvec=featvec,batch=BATCH_SIZE)
            centers_coor=tf.gather_nd(xyz,centers_id)
        for i in range(len(r_list)):
            r=r_list[i]
            k=k_list[i]
            group_id=point_choose.local_devide(sample_dim,k,r,xyz,centers_id,featvec,batch=BATCH_SIZE)
            group_coor=tf.gather_nd(xyz,group_id)
            group_coor=group_coor-tf.tile(tf.expand_dims(centers_coor,axis=2),multiples=[1,1,k,1])
            if featvec is not None:
                group_points=tf.gather_nd(featvec,group_id)
                if use_xyz:
                    group_points=tf.concat([group_points,group_coor],axis=-1)
            else:
                group_points=group_coor
            for j,out_channel in enumerate(layers_list[i]):
                group_points=conv2d(scope='conv%d_%d'%(i,j),inputs=group_points,num_outchannels=out_channel,kernel_size=[1,1],padding='VALID')
            newfeat=tf.reduce_max(group_points,axis=2)
            #newfeat=tf.squeeze(conv2d(scope='add_conv%d'%(i),inputs=group_points,num_outchannels=out_channel,kernel_size=[1,k],padding='VALID'))
            newfeatvec.append(newfeat)
    new_featvec_tensor=tf.concat(newfeatvec,axis=-1)
    return centers_coor,new_featvec_tensor
def mani_local_net(scope,sample_dim,xyz,featvec,global_feat,r_list,k_list,layers_list,trans_list,drop_list=None,use_feat=True,use_attention=False,use_mani=False,use_bnorm=False):
    with tf.variable_scope(scope):
        if use_feat and featvec is not None:
            input_tensor=tf.concat([xyz,featvec],axis=-1)
        else:
            input_tensor=xyz
        input_tensor=tf.expand_dims(input_tensor,axis=2)
        mani_tensor=input_tensor
        for i,outchannel in enumerate(trans_list):
            mani_tensor=conv2d('mani_layer%d'%i,mani_tensor,outchannel,[1,1],padding='VALID')
        mani_tensor=conv2d('mani_tensor',mani_tensor,3,[1,1],padding='VALID',activation_func=None)
        mani_tensor=tf.squeeze(mani_tensor,[2])
        if use_attention:
            idx,new_mani_xyz=attention_sample(sample_dim,mani_tensor,featvec,global_feat,[128,64,64])#center of manifold regions
            new_xyz=tf.reshape(tf.gather_nd(xyz,idx),[BATCH_SIZE,sample_dim,3])#corresponding raw centers in cartesian coordinates
        else:
            idx,new_mani_xyz=sampling(sample_dim,mani_tensor)
            new_xyz=tf.reshape(tf_sampling.gather_point(xyz,idx),[BATCH_SIZE,sample_dim,3])
        newfeatvec=[]
        if use_mani:
            groupfeat=tf.concat([mani_tensor,featvec],axis=-1)
        else:
            groupfeat=featvec 
        for i in range(len(r_list)):
            r=r_list[i]
            k=k_list[i]
            neighbors_idx,neighbors=grouping(xyz,new_xyz,mani_tensor,new_mani_xyz, r, k, groupfeat, knn=False, use_xyz=True)
            #print(neighbors)   
            for j,out_channel in enumerate(layers_list[i]):
                neighbors=conv2d(scope='conv%d_%d'%(i,j),inputs=neighbors,num_outchannels=out_channel,kernel_size=[1,1],padding='VALID')
                if use_bnorm:
                    neighbors=batch_normalization(neighbors,name='bnorm%d_%d'%(i,j))
                if drop_list is not None and drop_list[j]>0:
                    neighbors=dropout(neighbors,1-drop_list[j]) 
            newfeat=tf.reduce_max(neighbors,axis=2)
            newfeatvec.append(newfeat)
        new_featvec_tensor=tf.concat(newfeatvec,axis=-1)
    return new_xyz,new_featvec_tensor
def feat_interpolate(scope,xyz1,feat1,xyz2,feat2,mlp,use_bnorm):
    with tf.variable_scope('I'): 
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(feat2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, feat1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_pointsl=conv2d('featpro_layer%d'%i,new_pointsl,num_out_channel,[1,1],padding='VALID')
            if use_bnorm:
                neighbors=batch_normalization(neighbors,'bnorm%d'%i)  
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1 
def encoder(tensor):
    with tf.variable_scope('E'):
        ptnum=tensor.get_shape()[1].value
        #print(ptnum)
        global_feat=global_mlp(tensor,[64,128,256])
        #cen0,feat0=local_net(scope='layer0',sample_dim=2048,xyz=tensor,featvec=None,r_list=[0.02,0.05,0.1],k_list=[8,16,32],layers_list=[[16,16,32], [32,32,64], [64,64,128]],use_all=True)
        cen1,feat1=mani_local_net(scope='layer1',sample_dim=int(ptnum/2),xyz=tensor,featvec=None,global_feat=global_feat,r_list=[0.1,0.2,0.4],k_list=[32,64,128],
                                  layers_list=[[32,32,64], [64,64,128], [64,64,128]],trans_list=[64,64],drop_list=None,use_feat=True,use_attention=True,use_mani=False,use_bnorm=False)
        cen2,feat2=mani_local_net(scope='layer2',sample_dim=int(ptnum/8),xyz=cen1,featvec=feat1,global_feat=global_feat,r_list=[0.2,0.4,0.8],k_list=[32,64,128],
                                  layers_list=[[64,64,128], [128,128,256], [128,128,256]],trans_list=[64,64],drop_list=None,use_feat=True,use_attention=True,use_mani=False,use_bnorm=False)
        feat_concat=tf.expand_dims(tf.concat([feat2,cen2],axis=-1),axis=2)

        sum_net=conv2d('sumlayer1',feat_concat,256,[1,1],padding='VALID')#512
        sum_net=conv2d('sumlayer2',sum_net,512,[1,1],padding='VALID')#512
        sum_net=conv2d('sumlayer3',sum_net,1024,[1,1],padding='VALID')#512
        sum_net=tf.reduce_max(sum_net,axis=1) 
        sum_net=tf.squeeze(sum_net,[1])#512
        return sum_net
def raw_encoder(tensor):
    with tf.variable_scope('E'):
   # cen0,feat0=local_net(scope='layer0',sample_dim=2048,xyz=tensor,featvec=None,r_list=[0.02,0.05,0.1],k_list=[8,16,32],layers_list=[[16,16,32], [32,32,64], [64,64,128]],use_all=True)
        cen1,feat1=local_net(scope='layer1',sample_dim=512,xyz=tensor,featvec=None,r_list=[0.05,0.1,0.2,0.4],k_list=[16,32,64,128],layers_list=[[16,16,32],[32,32,64], [64,64,128], [64,64,128]])
        cen2,feat2=local_net(scope='layer2',sample_dim=128,xyz=cen1,featvec=feat1,r_list=[0.2,0.4,0.8],k_list=[32,64,128],layers_list=[[64,64,128], [128,128,256], [128,128,256]])
        feat_concat=tf.expand_dims(tf.concat([feat2,cen2],axis=-1),axis=2)

        full_net=tf.expand_dims(cen2,2)
        full_net=conv2d('full_layer1',full_net,64,[1,1],padding='VALID')
        full_net=conv2d('full_layer2',full_net,128,[1,1],padding='VALID')
        full_net = conv2d('full_layer3', full_net,512 , [1, 1], padding='VALID')
        full_net=tf.reduce_max(full_net,axis=1)
        #full_net=conv2d('add_layer1',full_net,128,[128,1],padding='VALID')
        full_net=tf.squeeze(full_net)
        #print(full_net)

        sum_net=conv2d('sumlayer1',feat_concat,512,[1,1],padding='VALID')#512
        sum_net=conv2d('sumlayer2',sum_net,512,[1,1],padding='VALID')#512
        sum_net=tf.reduce_max(sum_net,axis=1)
        #sum_net=conv2d('add_layer2',sum_net,512,[128,1],padding='VALID')
        sum_net=tf.squeeze(sum_net)#512
        #print(sum_net)
        sum_net=fully_connect('flayer1',tf.concat([sum_net,full_net],axis=-1),256)#256
        sum_net = fully_connect('flayer2', sum_net, 256)#256

        return sum_net

def classify(tensor):
    with tf.variable_scope('C'):
        net=fully_connect('flayer1',tensor,512)#256
        net=dropout(net,keep_prob=0.4)
        net=fully_connect('flayer2', net, 256)#256
        net=dropout(net,keep_prob=0.4)
        net=fully_connect('final_classify',net,40,activation_func=None)
    return net
def encoder_segment(point_cloud,is_training,num_class):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    xyz0 = point_cloud
    feat0 = None
    end_points['l0_xyz'] = l0_xyz
    
    global_feat=global_mlp(tensor,[64,64,128,512])
    xyz1,feat1=mani_local_net('seg_layer1',1024,xyz0,feat0,global_feat,[0.1],[32],[32,32,64],[64,64,3],drop_list=None,use_feat=True,use_attenion=False,use_mani=False,use_bnorm=False)
    xyz2,feat2=mani_local_net('seg_layer2',256,xyz1,feat1,global_feat,[0.2],[32],[64,64,128],[64,64,3],drop_list=None,use_feat=True,use_attenion=False,use_mani=False,use_bnorm=False)
    xyz3,feat3=mani_local_net('seg_layer3',64,xyz2,feat2,global_feat,[0.4],[32],[128,128,256],[64,64,3],drop_list=None,use_feat=True,use_attenion=False,use_mani=False,use_bnorm=False)
    xyz4,feat4=mani_local_net('seg_layer4',16,xyz3,feat3,global_feat,[0.8],[32],[256,256,512],[64,64,3],drop_list=None,use_feat=True,use_attenion=False,use_mani=False,use_bnorm=False)

    feat3=feat_interpolate('fp_layer1',xyz3,feat3,xyz4,feat4,[256,256],use_bnorm=True)
    feat2=feat_interpolate('fp_layer2',xyz2,feat2,xyz3,feat3,[256,256],use_bnorm=True)
    feat1=feat_interpolate('fp_layer3',xyz1,feat1,xyz2,feat2,[256,128],use_bnorm=True)
    feat0=feat_interpolate('fp_layer4',xyz0,feat0,xyz1,feat1,[128,128,128],use_bnorm=True)

    feat0=tf.expand_dims(feat0,axis=2)
    net=conv2d('fc_layer1',feat0,128,[1,1],padding='VALID')
    end_points['feats'] = tf.squeeze(net,[2])
    net=dropout(net,keep_prob=0.5,is_training=is_training,scope='dp1')
    net=conv2d('fc_layer2',net,num_class,[1,1],padding='VALID',activation_func=None)
    net=tf.squeeze(net,[2])
    return net,end_points
def global_mlp(scope,xyz,mlp,use_bnorm=False):
    with tf.variable_scope(scope):
        tensor=xyz
        for i,outchannel in enumerate(mlp):
            tensor=conv2d('ini_layer%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        result=tf.reduce_max(tensor,axis=1)
        result=tf.expand_dims(result,axis=1)
    return result
#batch*2048*1*128,batch*2048*1*128
def encode_cell(scope,input_tensor,state_tensor,state_len=128,code_len=128,mlp=[256,128],mlpout=[512,128],reuse=False,use_bnorm=False):
    with tf.variable_scope(scope,reuse=reuse):
        #input_info=conv2d('input_trans',input_tensor,code_len/2,[1,1],padding='VALID',use_bnorm=use_bnorm)#batch*2048*1*64
        #input_info=conv2d('input_trans1',input_info,code_len,[1,1],padding='VALID',use_bnorm=use_bnorm)#batch*2048*1*64
        input_info=input_tensor
        if state_tensor is not None:
            #new_state=conv2d('state_trans',state_tensor,128,[1,1],padding='VALID',use_bnorm=use_bnorm)#batch*2048*1*128
            state_info=tf.tile(state_tensor,multiples=[1,tf.shape(input_tensor)[1],1,1])#batch*2048*1*128
            new_state=tf.concat([input_info,state_info],axis=-1)#batch*2048*1*256
            #new_state=input_info*state_info
        else:
            new_state=input_info
        #new_state=conv2d('state_trans',new_state,128,[1,1],padding='VALID',use_bnorm=use_bnorm)#batch*2048*1*128
        #print(new_state,state_tensor,input_info)
        for i,outchannel in enumerate(mlp):
            new_state=conv2d('state%d'%i,new_state,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        new_state=tf.reduce_max(conv2d('state_end',new_state,state_len,[1,1],padding='VALID',use_bnorm=use_bnorm),axis=1)
        new_state=tf.expand_dims(new_state,axis=1)
        codeout=new_state
        for i,outchannel in enumerate(mlpout):
            codeout=conv2d('codemlp%d'%i,codeout,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        #codeout=tf.reduce_max(codeout,axis=1,keepdims=True)
    #batch*2048*1*128,batch*2048*1*128
    return codeout,new_state
def shuffle_points(prob,point):
    ptnum=point.get_shape()[1].value
    keepnum=int(PTNUM*prob)
    ptid=arange(PTNUM)
    batchid=tf.constant(value=arange(BATCH_SIZE),shape=[BATCH_SIZE,1,1],dtype=tf.int32)
    random.shuffle(ptid)
    idx=tf.constant(value=ptid[:keepnum],shape=[1,keepnum,1],dtype=tf.int32)
    idx=tf.concat([tf.tile(batchid,[1,keepnum,1]),tf.tile(idx,[BATCH_SIZE,1,1])],axis=-1)
    result=tf.gather_nd(point,idx)
    return result
#def model_substract(gt,partial_pts,min_dist=0.01,max_iter=3):
#    ptnum=gt.get_shape()[1].value
#    ptnum2=partial_pts.get_shape()[1].value
#    print(ptnum,ptnum2)
#    idxall=tf.tile(tf.reshape(tf.range(ptnum),[1,ptnum]),[tf.shape(gt)[0],1])#batch*npoint1
#    bidx=tf.tile(tf.reshape(tf.range(tf.shape(gt)[0]),[-1,1,1]),[1,ptnum2,1])#batch*npoint1*1
#    for i in range(max_iter):
#        dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(gt, partial_pts)#batch*npoint1,batch*npoint2
#        filter_dist=tf.where(tf.greater(dist2,min_dist),-tf.ones_like(dist2),tf.ones_like(dist2))
#        #print(filter_dist,idx2)
#        idx2_update=tf.reduce_max(idx2*tf.cast(filter_dist,tf.int32),axis=-1,keepdims=True)
#        idx2=tf.where(tf.greater(idx2,0),idx2,tf.tile(idx2_update,[1,ptnum2]))
#
#        update_idx=tf.concat([bidx,tf.expand_dims(idx2,axis=-1)])
#        update_data=-tf.ones_like(idx2)
#        idxtemp=tf.scatter_nd_update(idxall,update_idx,update_data)#batch*npoint1
#        replace_idx=tf.reduce_max(idxtemp,axis=-1,keepdims=True)
#        replace_idx=tf.tile(replace_idx,[1,ptnum2])
#        idxall=tf.scatter_nd_update(idxll,update_idx,replace_idx)
#    result_idx=tf.concat([bidx,tf.expand_dims(idxall,axis=-1)],axis=-1)#batch*npoint1*2
#    result=tf.gather_nd(gt,result_idx)#batch*npoint1*3
#    return result
def model_substract(gt,partial_pts,min_dist=10,max_iter=10,bsize=1):
    ptnum=gt.get_shape()[1].value
    ptnum2=partial_pts.get_shape()[1].value
    idxall=tf.Variable(tf.tile(tf.reshape(tf.range(ptnum),[1,ptnum]),[bsize,1]),name='subvar')#batch*npoint1
    bidx2=tf.tile(tf.reshape(tf.range(tf.shape(gt)[0]),[-1,1,1]),[1,ptnum2,1])#batch*npoint2*1
    bidx1=tf.tile(tf.reshape(tf.range(tf.shape(gt)[0]),[-1,1,1]),[1,ptnum,1])
    for i in range(max_iter):
        dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(gt, partial_pts)#batch*npoint1,batch*npoint2
        filter_dist=tf.where(tf.greater(dist2,min_dist),-tf.ones_like(dist2,dtype=tf.int32),tf.ones_like(dist2,dtype=tf.int32))
        idx2_update=tf.reduce_max(idx2*filter_dist,axis=-1,keepdims=True)
        idx2=tf.where(tf.greater(idx2,0),idx2,tf.tile(idx2_update,[1,ptnum2]))
        #print(idx2,bidx1)
        update_idx=tf.concat([bidx2,tf.expand_dims(idx2,axis=-1)],axis=-1)
        update_data=-tf.ones_like(idx2)
        idxtemp=tf.scatter_nd_update(idxall,update_idx,update_data)#batch*npoint1
        replace_idx=tf.tile(tf.reduce_max(idxtemp,axis=-1,keepdims=True),[1,ptnum2])
        #replace_idx=tf.tile(tf.constant(0,shape=[bsize,1]),[1,ptnum2])
        #print(idxtemp,replace_idx)
        #replace_idx=tf.tile(tf.constant(0,shape=[bsize,1]),[1,ptnum2])
        idxall=tf.scatter_nd_update(idxall,update_idx,replace_idx)
        result_idx=tf.concat([bidx1,tf.expand_dims(idxall,axis=-1)],axis=-1)#batch*npoint1*2

        gt=tf.gather_nd(gt,result_idx)#batch*npoint1*3

    #result_idx=tf.concat([bidx1,tf.expand_dims(idxall,axis=-1)],axis=-1)#batch*npoint1*2
    #result=tf.gather_nd(gt,result_idx)#batch*npoint1*3
    return gt
def naive_substract(gt, partial_pts, bsize=1):
    #print(gt)
    ptnum=gt.get_shape()[1].value
    ptnum2=partial_pts.get_shape()[1].value
    #idxall=tf.Variable(tf.tile(tf.reshape(tf.range(ptnum),[1,ptnum]),[bsize,1]),name='subvar')#batch*npoint1
    #print(tf.ones_like(gt))
    gtmul=tf.Variable(tf.ones([bsize,ptnum,1]),name='subvar',trainable=False)
    gtcp=gtmul
    bidx2=tf.tile(tf.reshape(tf.range(tf.shape(gt)[0]),[-1,1,1]),[1,ptnum2,1])#batch*npoint2*1
    #bidx1=tf.tile(tf.reshape(tf.range(tf.shape(gt)[0]),[-1,1,1]),[1,ptnum,1])

    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(gt, partial_pts)#batch*npoint1,batch*npoint2
    ptidx2=tf.concat([bidx2,tf.expand_dims(idx2,axis=-1)],axis=-1)#batch*npoint2*2
    #print(ptidx2)
    #ptidx2=tf.tile(tf.expand_dims(ptidx2,axis=2),[1,1,3,1])
    #print(ptidx2)
    #dimidx=tf.expand_dims(tf.tile(tf.reshape(tf.range(tf.shape(gt)[-1]),[1,1,-1]),[bsize,ptnum2,1]),axis=-1)
    #update_idx=tf.concat([ptidx2,dimidx],axis=-1)#batch*npoint*3*3
    #update_idx=tf.reshape(update_idx,[bsize,3*ptnum2,3])
    #print(update_idx)
    #assert False
    update_idx=ptidx2#tf.expand_dims(idx2,axis=-1)
    update_data=tf.cast(tf.reshape(tf.tile(tf.zeros_like(idx2),[1,1,1]),[-1,ptnum2,1]),tf.float32)
    #recover_data=tf.cast(tf.reshape(tf.tile(tf.ones_like(idx2),[1,1,1]),[-1,ptnum2,1]),tf.float32)

    result=tf.cast(tf.scatter_nd_update(gtmul,update_idx,update_data,use_locking=True),tf.float32)*gt
    #temp=tf.scatter_nd_update(gtmul,update_idx,update_data,use_locking=True)
    #gtmul=tf.scatter_nd_update(gtmul,update_idx,recover_data,use_locking=True)
    #result=tf.cast(temp,tf.float32)
    #temp=tf.cast(temp,tf.float32)
    #print(update_idx,update_data)
    #print(result) 
    return result

#seeds:batch*64*1*3
#output:batch*1024*1*3
def sec_loss(seeds,output,up_ratio=16):
    #seeds=tf.squeeze(seeds,[2])
    #outpts=tf.squeeze(output,[2])
    #outpts=tf.reshape(output,[-1,seeds.get_shape()[1].value*up_ratio,3])
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(seeds,output)

    bnum=tf.shape(seeds)[0]
    bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,output.get_shape()[1].value,1])
    idx=tf.concat([bid,tf.expand_dims(idx2,axis=-1)],axis=-1)
    match_seeds=tf.gather_nd(seeds,idx)

    matchvecs=output-match_seeds#batch*1024*3
    rawvecs=tf.reshape(tf.reshape(output,[-1,seeds.get_shape()[1].value,up_ratio,3])-tf.expand_dims(seeds,axis=2),[-1,seeds.get_shape()[1].value*up_ratio,3])#batch*1024*3
    print(rawvecs,matchvecs)
    result=tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(rawvecs-matchvecs),axis=-1)+1e-8))
    return result

    

def recover_cell(scope,input_tensor,con_tensor,mlp1,mlp2,mlpout=[128,128],reuse=False,use_bnorm=False):
    with tf.variable_scope(scope,reuse=reuse):
        tensor=input_tensor
        #for i,outchannel in enumerate(mlp1):
        #    tensor=conv2d('recover1%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        #tensor=tf.concat([tensor,input_tensor],axis=-1)
        tensor=tf.concat([tf.tile(tensor,multiples=[1,tf.shape(con_tensor)[1],1,1]),con_tensor],axis=-1)
        for i,outchannel in enumerate(mlp2):
            tensor=conv2d('recover2%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        tensor=tf.reduce_max(tensor,axis=1,keepdims=True)
        tensor=conv2d('recover2out%d'%i,tensor,mlp2[-1],[1,1],padding='VALID',use_bnorm=use_bnorm,activation_func=None)
        #tensor=tensor+input_tensor
        #tensor=tf.expand_dims(tensor,axis=1)
        #for i,outchannel in enumerate(mlpout):
        #    tensor=conv2d('codemlp%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
    return tensor
def recover_devide(scope,input_tensor,codelist,mlp=[256,256],state_len=128,use_bnorm=False):
    with tf.variable_scope(scope):
        codenum=len(codelist)
        tensor=tf.concat(codelist,axis=-1)
        tensor=tf.concat([tf.tile(tensor,multiples=[1,tf.shape(input_tensor)[1],1,1]),input_tensor],axis=-1)
        for i,outchannel in enumerate(mlp):
           tensor=conv2d('recover_devide%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        tensor=conv2d('recover_devideout',tensor,state_len*codenum,[1,1],padding='VALID',use_bnorm=use_bnorm)
        tensor=tf.reduce_max(tensor,axis=1,keepdims=True)
    return tensor[:,:,:,:128],tensor[:,:,:,128:256],tensor[:,:,:,256:]
#batch*2048*3
def re_encoder(pointcloud,rnum=3,type_pl='p'):
    with tf.variable_scope('E') as scope:
        #point=tf.expand_dims(pointcloud,axis=2)
        if type_pl=='r':
            prob=0.4*random.random()
            #prob=0.4
            point=shuffle_points(prob,pointcloud)
            #point=tf.nn.dropout(pointcloud,keep_prob=prob,noise_shape=[1,tf.shape(pointcloud)[1],1])*prob
            point_input=tf.expand_dims(point,axis=2)
        else:
            point=tf.expand_dims(pointcloud,axis=2)
            point_input=point
        statelen=128
        state0=global_mlp('init_mlp',point_input,[64,128,statelen])
        #point_input=recover_cell('recover1',point_input,mlp1=[128,256],mlp2=[512,1024])

        #state1=global_mlp('init_mlp1',point_input,[64,128,statelen])
        #state2=global_mlp('init_mlp2',point_input,[64,128,statelen])
       
        codeout1,state=encode_cell('cell',point_input,state0,mlp=[256,384],mlpout=[128,128],state_len=statelen,code_len=128,use_bnorm=False)
        codeout1=recover_cell('recover1',codeout1,point_input,mlp1=[256,256],mlp2=[128,128])
        codeout2,state=encode_cell('cell',point_input,state,mlp=[256,384],mlpout=[128,128],state_len=statelen,code_len=128,reuse=True,use_bnorm=False)
        codeout2=recover_cell('recover2',codeout2,point_input,mlp1=[256,256],mlp2=[128,128])
        codeout3,state=encode_cell('cell',point_input,state,mlp=[256,384],mlpout=[128,128],state_len=statelen,code_len=128,reuse=True,use_bnorm=False)
        codeout3=recover_cell('recover3',codeout3,point_input,mlp1=[256,256],mlp2=[128,128])
        #codeout1,codeout2,codeout3=recover_devide('recover_final',point_input,[codeout1,codeout2,codeout3],mlp=[256,256],state_len=128)
        #state=global_mlp('end_mlp',point_input,[64,128,128])
        #codeout4,state=encode_cell('cell',point_input,state,mlp=[256,128],reuse=True,use_bnorm=False)
        tf.add_to_collection('code1', codeout1)
        tf.add_to_collection('code2', codeout2)
        tf.add_to_collection('code3', codeout3)
        #tf.add_to_collection('code4', codeout4)
    return 0
def merge_layer(rawpts,newpts,decfactor,knum=16):
    npoint=newpts.get_shape()[1].value
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(rawpts, newpts)
    #_,idx = tf_grouping.knn_point(knum, rawpts, newpts)
    #else:
    #    idx,_ = tf_grouping.query_ball_point(radius, ksample, rawpts, cenpts)
    #print(idx2)
    grouped_xyz = tf_grouping.group_point(rawpts, tf.expand_dims(idx2,axis=-1)) # (batch_size, npoint_newpts, knum, 3)
    #bid=tf.tile(tf.reshape(tf.range(start=0,limit=tf.shape(rawpts)[0],dtype=tf.int32),[-1,1,1,1]),[1,npoint,knum,1])
    #idx=tf.concat([bid,tf.expand_dims(idx,axis=-1)],axis=-1)
    #grouped_xyz=tf.gather_nd(rawpts,idx)

    dismat=tf.reduce_sum(tf.square(grouped_xyz-tf.expand_dims(newpts,axis=2)),axis=-1,keepdims=True)
    #ratio=tf.exp(-dismat/(1e-8+tf.square(decfactor)))/(1e-8+tf.reduce_sum(tf.exp(-dismat/(1e-8+tf.square(decfactor))),axis=2,keepdims=True))
    ratio=tf.exp(-dismat/(1e-8+tf.square(decfactor)))
    refine_pts=newpts+tf.reduce_sum(ratio*(grouped_xyz-tf.expand_dims(newpts,axis=2)),axis=2)
    return refine_pts
def pt_mlp(scope,xyz,mlp,use_bnorm=False):
    with tf.variable_scope(scope):
        tensor=tf.expand_dims(xyz,axis=2)
        for i,outchannel in enumerate(mlp):
            tensor=conv2d('ini_layer%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        result=tf.reduce_max(tensor,axis=1,keepdims=True)
        #result=tf.expand_dims(result,axis=1)
    return tensor,result
#def init_layer(l0_xyz,codeword,mlp=[256,256,256],state_len=128,is_training=True,bn_decay=0.95,use_bnorm=False):
#    _,gfeat0=pt_mlp('pt_mlp0',l0_xyz,[128,128])
#    transform = my_transform_net(l0_xyz,tf.concat([codeword,gfeat0],axis=-1) ,is_training, bn_decay, K=3)
#    xyz_transformed = tf.matmul(l0_xyz, transform)#batch*128*3
#    xyz=tf.concat([l0_xyz,xyz_transformed],axis=1)
#    tensor,gfeat=pt_mlp('pt_mlp',xyz,[128,128])
#    gfeat=tf.concat([gfeat,codeword],axis=-1)
#    tensor=tf.concat([tf.expand_dims(xyz,axis=2),tf.tile(gfeat,[1,tf.shape(xyz)[1],1,1])],axis=-1)
#    for i,outchannel in enumerate(mlp):
#        tensor=conv2d('init_layer%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
#    featresult=conv2d('init_feat_out',tensor,state_len,[1,1],padding='VALID',use_bnorm=use_bnorm,activation_func=tf.nn.relu)
#    result=conv2d('init_layer_out',tensor,3,[1,1],padding='VALID',use_bnorm=use_bnorm,activation_func=None)
#    return tf.squeeze(result,[2]),featresulti
def init_layer(l0_xyz,codeword,mlp0=[256,256],mlp1=[128,64],mlp2=[256,256],mlp3=[256,256],state_len=128,is_training=True,bn_decay=0.95,use_bnorm=False):
    _,gfeat0=pt_mlp('pt_mlp0',l0_xyz,[128,128])
    #transform = my_transform_net(l0_xyz,tf.concat([codeword,gfeat0],axis=-1) ,is_training, bn_decay, K=3)
    ptnum=l0_xyz.get_shape()[1].value
    tensor=tf.concat([tf.expand_dims(l0_xyz,axis=2),tf.tile(gfeat0,[1,ptnum,1,1])],axis=-1)
    for i,outchannel in enumerate(mlp0):
        tensor=conv2d('trans_layer0%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
    maxtensor=tf.reduce_max(tensor,axis=1,keepdims=True)
    maxtensor=tf.concat([tf.expand_dims(l0_xyz,axis=2),tf.tile(maxtensor,[1,tf.shape(l0_xyz)[1],1,1])],axis=-1)
    for i,outchannel in enumerate(mlp1):
        maxtensor=conv2d('trans_layer1%d'%i,maxtensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
    transmat=conv2d('trans_layer1out%d'%i,maxtensor,12,[1,1],padding='VALID',use_bnorm=use_bnorm,activation_func=None)
    transform,move=tf.reshape(transmat[:,:,:,:9],[-1,ptnum,3,3]),transmat[:,:,:,9:]

    xyz_transformed = tf.matmul(tf.expand_dims(l0_xyz,axis=2), transform)+move#batch*128*3
    xyz_transformed=tf.squeeze(xyz_transformed,[2])
    xyz=tf.concat([l0_xyz,xyz_transformed],axis=1)

    _,gfeat=pt_mlp('pt_mlp',xyz,[128,128])
    gfeat=tf.concat([gfeat,codeword],axis=-1)
    tensor=tf.concat([tf.expand_dims(xyz,axis=2),tf.tile(gfeat,[1,tf.shape(xyz)[1],1,1])],axis=-1)
    for i,outchannel in enumerate(mlp2):
        tensor=conv2d('init_layer2%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
    maxtensor=tf.reduce_max(tensor,axis=1,keepdims=True)
    maxtensor=tf.concat([tf.expand_dims(xyz,axis=2),tf.tile(maxtensor,[1,tf.shape(xyz)[1],1,1])],axis=-1)

    for i,outchannel in enumerate(mlp3):
        maxtensor=conv2d('trans_layer3%d'%i,maxtensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
    movemat=conv2d('trans_layer3out%d'%i,maxtensor,state_len+3,[1,1],padding='VALID',use_bnorm=use_bnorm,activation_func=None)
    featresult=tf.nn.relu(movemat[:,:,:,:-3])
    result=xyz+tf.squeeze(movemat[:,:,:,-3:],axis=2)
    #featresult=conv2d('init_feat_out',tensor,state_len,[1,1],padding='VALID',use_bnorm=use_bnorm,activation_func=tf.nn.relu)
    #result=conv2d('init_layer_out',tensor,3,[1,1],padding='VALID',use_bnorm=use_bnorm,activation_func=None)
    return result,featresult
def init_move_layer(startpts,codeword,mlp=[256,256,256],mlp1=[256,128],mlp2=[256,128,64],state_len=128,use_bnorm=False):
    tensor0=tf.expand_dims(startpts,axis=2)
    ptnum=startpts.get_shape()[1].value
    codelen=codeword.get_shape()[-1].value
    tensor1=tf.concat([tensor0,tf.tile(tf.reshape(codeword,[-1,1,1,codelen]),[1,ptnum,1,1])],axis=-1)
    tensor=tensor1
    for i,outchannel in enumerate(mlp):
        tensor=conv2d('ini_layer%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
    maxtensor=tf.reduce_max(tensor,axis=1,keepdims=True)
    tensor=tf.concat([tensor1,tf.tile(maxtensor,[1,ptnum,1,1])],axis=-1)
    outfeats=tensor
    for i,outchannel in enumerate(mlp1):
        outfeats=conv2d('ini_featout%d'%i,outfeats,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
    outfeats=conv2d('inimove_featout',outfeats,state_len,[1,1],padding='VALID',use_bnorm=use_bnorm,activation_func=tf.nn.relu)
    for i,outchannel in enumerate(mlp2):
        tensor=conv2d('ini_ptsout%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
    outpts=conv2d('inimove_ptsout',tensor,3,[1,1],padding='VALID',use_bnorm=use_bnorm,activation_func=tf.nn.tanh)
    outpts=startpts+tf.squeeze(outpts,axis=[2])
    #outfeats=tf.squeeze(outfeats,axis=[2])
    return outpts,outfeats
def feat_trans(feat,mlp=[256,256],use_bnorm=False):
    for i,outchannel in enumerate(mlp):
        feat=conv2d('partfeat%d'%i,feat,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
    return feat
def get_topk(rawcode,codepool,knum):
    valdist,ptid = tf_grouping.knn_point(knum, codepool, rawcode)#batch*n*k
    bid=tf.tile(tf.reshape(tf.range(start=0,limit=tf.shape(rawcode)[0],dtype=tf.int32),[-1,1,1,1]),[1,tf.shape(rawcode)[1],knum,1])
    idx=tf.concat([bid,tf.expand_dims(ptid,axis=-1)],axis=-1)
    #kcode=tf.gather_nd(codepool,idx)#batch*n*k*c
    #kdist=tf.reduce_mean(tf.square(tf.expand_dims(rawcode,axis=2)-kcode),axis=-1)
    return idx
#cens;batch*64*3
#pts:batch*(N+64)*3
#gfeat:batch*1*1*featlen
def knn_cell(scope,cens,pts,mlp=[64,128,128],gfeat=None,knum=32,use_resi=False,use_bnorm=False):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        #feat=tf.expand_dims(pts,axis=2)
        feat=pts
        ptnum=pts.get_shape()[1].value
        cennum=cens.get_shape()[1].value
        statelen=128
        if gfeat is not None:
            feat=tf.concat([feat,tf.tile(tf.squeeze(gfeat,[2]),[1,ptnum,1])],axis=-1)
            statelen=gfeat.get_shape()[-1].value
        idx=get_topk(cens,pts,knum=knum)
        kfeat=tf.gather_nd(feat,idx)-tf.expand_dims(cens,axis=2)#batch*n*k*c
        for i,outchannel in enumerate(mlp):
            kfeat=conv2d('localfeat%d'%i,kfeat,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        if gfeat is not None and use_resi:
            kfeat=conv2d('localfeatout',kfeat,statelen,[1,1],padding='VALID',use_bnorm=use_bnorm,activation_func=None)#batch*cennum*k*statelen
            kfeat=kfeat+tf.tile(gfeat,[1,cennum,knum,1])
        else:
            kfeat=conv2d('localfeatout',kfeat,statelen,[1,1],padding='VALID',use_bnorm=use_bnorm,activation_func=tf.nn.relu)#batch*cennum*k*statelen
        result=tf.reduce_max(kfeat,axis=2,keepdims=True)
    return result
def re_chamfer(gt,pred,part=8):
    ptnum=gt.get_shape()[1].value
    interval=int(ptnum/8)
    #gt=shuffle_points(gt)
    bnum=tf.shape(pred)[0]
    ptnum=pred.get_shape()[1].value
    ptids=arange(ptnum)
    #random.shuffle(ptids)
    ptids=tf.constant(ptids,tf.int32)
    #ptids=arange(ptnum)
    #ptids=tf.random_shuffle(ptids,seed=None)
    emdlist=[]
    for i in range(part):
        ptid=tf.cast(tf.tile(tf.reshape(ptids[i*interval:(i+1)*interval],shape=[1,interval,1]),[bnum,1,1]),tf.int32)
        bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,interval,1])
        idx=tf.concat([bid,ptid],axis=-1)
        new_gt=tf.gather_nd(gt,idx)
        new_xyz=tf.gather_nd(pred,idx)

        #new_gt=gt[:,i*interval:(i+1)*interval,:]
        emdlist.append(chamfer_big(new_xyz,new_gt)[0])
    loss=sum(emdlist)/part
    return loss
def get_repulsion_loss4(cens,pred, nsample=16, radius=0.07):
    # pred: (batch_size, npoint,3)
    idx, pts_cnt = query_ball_point(radius, nsample, pred, cens)
    #tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(cens, 2)

    ##get the uniform loss
    h = 0.03
    dist_square = tf.reduce_sum(grouped_pred ** 2, axis=-1)
    dist_square, idx = tf.nn.top_k(-dist_square, 5)
    dist_square = -dist_square[:, :, 1:]  # remove the first one
    dist_square = tf.maximum(1e-12, dist_square)
    dist = tf.sqrt(dist_square)
    weight = tf.exp(-dist_square / h ** 2)
    uniform_loss = tf.reduce_mean(radius - dist * weight)
    return uniform_loss
#codeword:batch*1*1*codelen
#codepool:1*n*1*codelen
#decayratio:(1,)
def add_prior(codeword,codepool,decayratio):
    codepool=tf.tile(codepool,[tf.shape(codeword)[0],1,1,1])
    kdist=tf.reduce_mean(tf.square(codeword-codepool),axis=-1,keepdims=True)
    kmask=tf.exp(-kdist/(1e-8+decayratio))/(1e-8+tf.reduce_sum(tf.exp(-kdist/(1e-8+decayratio)),axis=-1,keepdims=True))
    kcode=tf.reduce_sum(kmask*kdist,axis=1,keepdims=True)
    return kcode
def get_priorpool(shape,poolnum=3,featlen=256):
    pool_list=[]
    for i in range(poolnum):
        name='pool_'+str(i)
        pool=tf.get_variable(name=name,shape=shape,initializer=tf.contrib.layers.xavier_initializer())
        pool_list.append(pool)
    return pool_list
#ptsin:batch*3000*3
#feat:batch*1*1*len
def sp_layer(ptsin,feat,mlp0=[128,64],mlp=[64,128],mlp1=[64,64,3],featlen=128,relen=8,ptnum=64,use_bn=False):
    pts=tf.expand_dims(ptsin,axis=2)
    k=int(ptnum/relen)
    net=tf.concat([pts,tf.tile(feat,[1,tf.shape(pts)[1],1,1])],axis=-1)
    for i,outchannel in enumerate(mlp0):
        net=conv2d('spnet%d'%i,net,outchannel,[1,1],padding='VALID',use_bnorm=use_bn)
    net=conv2d('spfeat',net,relen,[1,1],padding='VALID',use_bnorm=use_bn)#batch*3000*1*relen
    snet=tf.squeeze(net,2)#batch*3000*relen
    snet=tf.nn.softmax(snet,axis=-1)
    stnet=tf.transpose(snet,[0,2,1])

    _, idx = tf.nn.top_k(stnet, k)#batch*relen*k
    bid=tf.tile(tf.reshape(tf.range(start=0,limit=tf.shape(pts)[0],dtype=tf.int32),[-1,1,1,1]),[1,relen,k,1])
    idx=tf.concat([bid,tf.expand_dims(idx,axis=-1)],axis=-1)
    spnet=tf.gather_nd(snet,idx)#batch*relen*k*relen

    mlp.append(featlen)
    for i,outchannel in enumerate(mlp):
        spnet=conv2d('spnet_feat%d'%i,spnet,outchannel,[1,3],padding='SAME',use_bnorm=use_bn)#batch*relen*k*featlen
    myfeat=tf.reshape(spnet,[-1,relen*k,1,featlen])

    for i,outchannel in enumerate(mlp1):
        spnet=conv2d('spnet_pts%d'%i,spnet,outchannel,[1,1],padding='VALID',use_bnorm=use_bn)
    spnet=conv2d('spfeat_ptsout',spnet,3,[1,1],padding='VALID',use_bnorm=use_bn,activation_func=tf.nn.tanh)#batch*ptnum*1*3
    mypts=tf.reshape(spnet,[-1,relen*k,3])
    return mypts,myfeat
def mlp_conv(inputs, layer_dims, bn=None, bn_params=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv1d(
            inputs, num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='conv_%d' % i)
    outputs = tf.contrib.layers.conv1d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope='conv_%d' % (len(layer_dims) - 1))
    tf.add_to_collection('refine_layer_final16384',outputs)
    return outputs    
def DP_refine(coord_feature,coarse_highres,gl_feature):
    ptnum=coarse_highres.get_shape()[1].value
    fps_idx,coarse_fps=sampling(int(ptnum)/2,coarse_highres,use_type='f')

    bid=tf.tile(tf.reshape(tf.range(start=0,limit=tf.shape(fps_idx)[0],dtype=tf.int32),[-1,1,1]),[1,int(ptnum/2),1])
    idx=tf.concat([bid,tf.expand_dims(fps_idx,axis=-1)],axis=-1)
    print('.......',coord_feature)
    coord_feature=tf.gather_nd(coord_feature,idx)

    #bnum=tf.shape(coord_feature)[0]
    #bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,int(self.num_fine/2),1])
    #idx=tf.concat([bid,tf.expand_dims(fps_idx,axis=-1)],axis=-1)
    #coarse_fps=tf.gather_nd(coarse_highres,idx)
    print('...................',coord_feature)
    #coord_feature = gather_point(coord_feature, fps_idx)
    #coarse_fps = gather_point(coarse_highres, fps_idx)

    #coord_feature = tf.expand_dims(coord_feature, 2)

    #print('coord_feature', coord, coord_feature)

    score = conv2d('fc_layer3',coord_feature, 16, [1, 1],padding='VALID', stride=[1, 1],use_bnorm=False)
    score = conv2d('fc_layer4',score, 8, [1, 1],padding='VALID', stride=[1, 1],use_bnorm=False)
    score = conv2d('fc_layer5',score, 1, [1, 1],padding='VALID', stride=[1, 1], use_bnorm=False)

    score = tf.nn.softplus(score)
    score = tf.squeeze(score, [2,3])

    gridsize=2
    _, idx = tf.math.top_k(score, int(ptnum/(gridsize**2)))

    #coarse = gather_point(coarse_fps, idx)

    coord_feature = tf.squeeze(coord_feature, [2])

    bnum=tf.shape(coord_feature)[0]
    bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,int(ptnum/(gridsize**2)),1])
    idx=tf.concat([bid,tf.expand_dims(idx,axis=-1)],axis=-1)
    coarse=tf.gather_nd(coarse_fps,idx)

    #bnum=tf.shape(coord_feature)[0]
    #bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,self.num_coarse,1])
    #idx=tf.concat([bid,tf.expand_dims(idx,axis=-1)],axis=-1)
    coord_feature=tf.gather_nd(coord_feature,idx)
    #coord_feature = gather_point(coord_feature, idx)

    print('coarse', coord_feature, coarse)
    with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
            grid = tf.meshgrid(tf.linspace(-0.05, 0.05, gridsize), tf.linspace(-0.05, 0.05, gridsize))
            print('grid:', grid)
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            print('grid:', grid)
            grid_feat = tf.tile(grid, [bnum, int(ptnum/(gridsize**2)), 1])
            print('grid_feat', grid_feat)

            point_feat = tf.tile(tf.expand_dims(tf.concat([coarse, coord_feature], axis=-1), 2), [1, 1, gridsize ** 2, 1])
            point_feat = tf.reshape(point_feat, [-1, ptnum, point_feat.get_shape()[-1].value])
            print('point_feat', point_feat)

            global_feat = tf.tile(tf.squeeze(gl_feature, 1), [1, ptnum, 1])

            #print('global_feat', global_feat)

            feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)
            print('feat:', feat)

            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, gridsize**2, 1])
            center = tf.reshape(center, [-1, ptnum, 3])

            print('center', center)

            fine = mlp_conv(feat, [256, 256, 3]) + center
            print('fine:', fine)
    return fine


def full_process(pointcloud,type_pl='p'):
    if type_pl=='r':
        prob=0.4*random.random()
        #prob=0.4
        point=shuffle_points(prob,pointcloud)
        #point=tf.nn.dropout(pointcloud,keep_prob=prob,noise_shape=[1,tf.shape(pointcloud)[1],1])*prob
        point_input=tf.expand_dims(point,axis=2)
    else:
        point=tf.expand_dims(pointcloud,axis=2)
        point_input=point
    ptnum=16384
    statelen=256
    state0=global_mlp('init_mlp',point_input,[64,128,statelen])
    
    codeout1,state=encode_cell('cell',point_input,state0,mlp=[256,384],mlpout=[256,256],state_len=statelen,code_len=256,use_bnorm=False)

    #pool1,pool2,pool3=get_priorpool([1,32,1,codeout1.get_shape()[-1].value],poolnum=3)

    #codeout1=recover_cell('recover1',add_prior(codeout1,pool1,decayratio=0.01),point_input,mlp1=[256,256],mlp2=[256,256],use_bnorm=False)
    codeout1=recover_cell('recover1',codeout1,point_input,mlp1=[256,256],mlp2=[256,256],use_bnorm=False)

    #ptspool=tf.get_variable(name='points_var',shape=[1,64,8,3],initializer=tf.contrib.layers.xavier_initializer()),[tf.shape(points3)[0],1,1]
    #pts1=tf.tile(ptspool,[tf.shape(point_input)[0],1,1,1])
    #pts1=tf.tile(tf.get_variable(name='points_var',shape=[1,64,3],initializer=tf.contrib.layers.xavier_initializer()),[tf.shape(point_input)[0],1,1])
    #points1,dstate=init_move_layer(pts1,codeout1,state_len=128,use_bnorm=False)
    #tf.add_to_collection('startvar', pts1)

    #points1,dstate=sp_layer(pointcloud,codeout1,mlp0=[128,64],mlp=[64,128],mlp1=[128,64,64],featlen=128,relen=8,ptnum=64,use_bn=False)

    #points1,dstate=init_layer(sampling(32,pointcloud,use_type='f')[1],codeout1,mlp0=[256,256],mlp1=[128,64],mlp2=[256,256],mlp3=[256,256],state_len=128,is_training=True,bn_decay=0.95,use_bnorm=False)
    #points0,dstate0=init_decode_layer('init_cell',codeout1,None,ptnum=32,mlp=[256,256],mlp1=[128],mlp2=[256,256],state_len=128,use_bnorm=False)#batch*128*3,batch*128*1*218
    points1,dstate=init_move_layer(sampling(32,pointcloud,use_type='f')[1],codeout1,state_len=128,use_bnorm=False)
    partfeat=global_mlp('part_mlp',tf.expand_dims(tf.concat([pointcloud,points1],axis=1),axis=2),[64,128,statelen])
    #partfeat=global_mlp('part_mlp',point_input,[64,128,statelen])
    points0,dstate0=init_decode_layer('init_cell',feat_trans(tf.concat([partfeat,codeout1],axis=-1)),None,ptnum=32,mlp=[256,256],mlp1=[128],mlp2=[256,256],state_len=128,use_bnorm=False)#batch*128*3,batch*128*1*218
    points1,dstate=tf.concat([points0,points1],axis=1),tf.concat([dstate0,dstate],axis=1)
    #points1,dstate=init_layer(sampling(32,pointcloud,use_type='f')[1],codeout1,mlp=[256,256,256],state_len=128,is_training=True,bn_decay=0.95)
    #points1=tf.nn.tanh(points1)


    tf.add_to_collection('points1', points1)
    #points1,dstate=refine_layer('refine_layer',points1,feat=tf.concat([tf.get_collection('code1')[0],tf.get_collection('code2')[0]],axis=-1),feat2=dstate,mlpself=None)

    decfactor0=tf.get_variable(name='decline_factor0',shape=[1],initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection('decfactor',tf.square(decfactor0))
    points1=merge_layer(pointcloud,points1,decfactor0,knum=1)
    #points1,dstate=refine_layer('refine_layer1',points1,feat=codeout1,mlp=[128,64,64],mlp2=[128,128],mlpself=[128,128],feat2=dstate,use_bnorm=False)
    points1,dstate=refine_layer('refine_layer1',points1,feat=codeout1,feat2=dstate,use_bnorm=False)

    #print(point_input,points1)
    point_input=tf.concat([tf.expand_dims(pointcloud,axis=2),tf.expand_dims(points1,axis=2)],axis=1)
    #kfeat=knn_cell('knn1',points1,tf.squeeze(point_input,[2]),mlp=[64,128,128],gfeat=None,knum=32,use_resi=False,use_bnorm=False)
    #dstate=kfeat
    #out1=tf.squeeze(point_input,axis=[2])
    codeout2,state=encode_cell('cell',point_input,state,mlp=[256,384],mlpout=[256,256],state_len=statelen,code_len=256,reuse=True,use_bnorm=False)
    #codeout2=recover_cell('recover2',add_prior(codeout2,pool2,decayratio=0.01),point_input,mlp1=[256,256],mlp2=[256,256],use_bnorm=False)
    codeout2=recover_cell('recover2',codeout2,point_input,mlp1=[256,256],mlp2=[256,256],use_bnorm=False)
    codeout2=codeout1+codeout2
    points2,dstate=decode_cell('decode_cell',codeout2,points1,dstate,None,mlp1=[128,64],up_ratio=16,state_len=128,reuse=False,use_bnorm=False)
    #points2,dstate=fn_decode_cell('decode_cell',codeout2,points1,dstate,lastcode=None,up_ratio=16,mlp=[256,256],mlp_trans=[128,64],mlp_transmat=[64,9],mlp_grid=[128,64,3],\
    #               mlp_mask=[128,128],mlpfn0=[64,64],mlpfn1=[64,32],mlpfn2=[64,32],mlp2=[128,128],grid_scale=0.05,state_len=128,reuse=False)
    tf.add_to_collection('points2', points2)
    #print(points2)
    #tf.add_to_collection('sec2', sec_loss(points1,points2,up_ratio=16))
    #points2,dstate=refine_layer('refine_layer',points2,feat=tf.concat([tf.get_collection('code2')[0],tf.get_collection('code3')[0]],axis=-1),feat2=dstate,mlpself=None,reuse=True)

    decfactor1=tf.get_variable(name='decline_factor1',shape=[1],initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection('decfactor',tf.square(decfactor1))
    points2=merge_layer(pointcloud,points2,decfactor1,knum=1)
    #points2,dstate=refine_layer('refine_layer1',points2,feat=codeout2,mlp=[128,64,64],mlp2=[128,128],mlpself=[128,128],feat2=dstate,reuse=True,use_bnorm=False)
    points2,dstate=refine_layer('refine_layer2',points2,feat=codeout2,feat2=dstate,reuse=False,use_bnorm=False)

    point_input=tf.concat([tf.expand_dims(pointcloud,axis=2),tf.expand_dims(points2,axis=2)],axis=1)
    #kfeat=knn_cell('knn2',points2,tf.squeeze(point_input,[2]),mlp=[64,128,128],gfeat=None,knum=32,use_resi=False,use_bnorm=False)
    #dstate=kfeat
    #out2=tf.squeeze(point_input,axis=[2])
    codeout3,state=encode_cell('cell',point_input,state,mlp=[256,384],mlpout=[256,256],state_len=statelen,code_len=256,reuse=True,use_bnorm=False)
    #codeout3=recover_cell('recover3',add_prior(codeout3,pool3,decayratio=0.01),point_input,mlp1=[256,256],mlp2=[256,256],use_bnorm=False)
    codeout3=recover_cell('recover3',codeout3,point_input,mlp1=[256,256],mlp2=[256,256],use_bnorm=False)
    codeout3=codeout2+codeout3
    points3,dstate=decode_cell('decode_cell',codeout3,points2,dstate,None,mlp1=[128,64],up_ratio=16,state_len=128,reuse=True,use_bnorm=False)
    #points3,dstate=fn_decode_cell('decode_cell',codeout3,points2,dstate,lastcode=None,up_ratio=16,mlp=[256,256],mlp_trans=[128,64],mlp_transmat=[64,9],mlp_grid=[128,64,3],\
    #               mlp_mask=[128,128],mlpfn0=[64,64],mlpfn1=[64,32],mlpfn2=[64,32],mlp2=[128,128],grid_scale=0.05,state_len=128,reuse=True)

    #ptsin=sampling(128,points3,use_type='f')[1]
    #point_input=tf.concat([tf.expand_dims(pointcloud,axis=2),tf.expand_dims(ptsin,axis=2)],axis=1)
    #point_input=sampling(1024,pointcloud,use_type='f')[1]
    #codeout4,state=encode_cell('cell',point_input,state,mlp=[256,384],mlpout=[128,128],state_len=statelen,code_len=128,reuse=True,use_bnorm=False)
    #codeout4=recover_cell('recover4',codeout4,point_input,mlp1=[256,256],mlp2=[128,128],use_bnorm=True)
    #out3=tf.concat([pointcloud,points3],axis=1)
    points_final=points3
    #tf.add_to_collection('sec3', sec_loss(points2,points3,up_ratio=16))

    #points_var=tf.tile(tf.get_variable(name='points_var',shape=[1,16384,3],initializer=tf.contrib.layers.xavier_initializer()),[tf.shape(points3)[0],1,1])
    #decfactor2=tf.get_variable(name='decline_factor2',shape=[1],initializer=tf.contrib.layers.xavier_initializer())
    #tf.add_to_collection('decfactor',tf.square(decfactor2))
    #tf.add_to_collection('varpts',points_var)

    #for i in range(3):
    decfactor=tf.get_variable(name='decline_factor',shape=[1],initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection('decfactor',tf.square(decfactor))


    #_,pointcloud=sampling(128,pointcloud,use_type='f')
    points_final=merge_layer(pointcloud,points_final,decfactor,knum=1)
    #points_final=merge_layer(points_final,points_var,decfactor2,knum=1)
    #points_final=merge_layer(pointcloud,points_final,decfactor,knum=1)
    #points_final=merge_layer(points_var,points_final,decfactor2,knum=1)
    #print(points_final)
    #points3,state=refine_layer('refine_layer',points3,feat=tf.concat([tf.get_collection('code2')[0],tf.get_collection('code3')[0]],axis=-1),feat2=state,mlpself=None,reuse=True)
    points_final,state=refine_layer('refine_layer_final',points_final,feat=codeout3,feat2=dstate,reuse=False,use_bnorm=False)
    #points_final,state=refine_layer('refine_layer_final',points_final,feat=codeout3,mlp=[128,64,64],mlp2=[128,128],mlpself=[128,128],feat2=dstate,reuse=False,no_feat=True,use_bnorm=False)

    #points_final=DP_refine(dstate,points_final,codeout3)

    #decfactor2=tf.get_variable(name='decline_factor2',shape=[1],initializer=tf.contrib.layers.xavier_initializer())
    #f.add_to_collection('decfactor',tf.square(decfactor2))

    #_,pointcloud=sampling(128,pointcloud,use_type='f')
    #points_final=merge_layer(pointcloud,points_final,decfactor2,knum=1)
    #points_final,state=refine_layer('refine_layer_final2',points_final,feat=codeout4,mlp=[128,128,128],feat2=None,mlpself=None,reuse=False,no_feat=True,use_bnorm=False)
    #points_final=tf.concat([pointcloud,points3],axis=1)
    #print(points_final)
    #_,points_final=sampling(ptnum,points_final,use_type='f')
    #print(points_final)
    #points_final=res_layer('res_final',points_final,mlp=[64,128,256],mlp2=[256,128,64])
    tf.add_to_collection('o2048', points3)
    tf.add_to_collection('code1', codeout1)
    tf.add_to_collection('code2', codeout2)
    tf.add_to_collection('code3', codeout3)
    #tf.add_to_collection('final', points_final)
    return points1,points2,points3,points_final
def multiple_calculate(scope,codelist,mlp=[128,128,64,5],ptnum=16384):
    with tf.variable_scope(scope):
        digit=tf.log(ptnum)/tf.log(2)
        codes=tf.concat(codelist,axis=-1)
        for i,outchannel in enumerate(mlp2):
            codes=conv2d('state%d'%i,codes,outchannel,[1,1],padding='VALID')
        #init_digits=[tf.cast(codes[:,:,:,0],tf.int32),tf.cast(codes[:,:,:,1],tf.int32)]
        init_digits=[tf.cast(tf.reduce_mean(codes[:,:,:,0]),tf.int32),tf.cast(tf.reduce_mean(codes[:,:,:,1]),tf.int32)]
        
        init_size=[codes[:,:,:,2],codes[:,:,:,3]]#two batch*1*1*1
        #multi1=codes[:,:,:,4]
        multi1=tf.cast(tf.reduce_mean(codes[:,:,:,4]),tf.int32)
        multi2=digit-init_digits[0]-init_digits[1]-multi1
        multi1,multi2=tf.maximum(multi1,1),tf.maximum(multi2,1)
        init_digits[0]=tf.minimum(init_digits[0],digit-multi1-multi2)
        init_digits[1]=digit-multi1-multi2-init_digits[0]
        #multi2=digit-init_digits[0]-init_digits[1]-multi1
        init_grid=[tf.pow(2,init_digits[0]),tf.pow(2,init_digits[1])]#two batch*1*1*1
    return init_grid,init_size,tf.pow(2,multi1),tf.pow(2,multi2)
def init_memory(scope,tensor_shape):
    with tf.variable_scope(scope):
        memory=tf.get_variable(name='memory_capsule',shape=tensor_shape,initializer=tf.contrib.layers.xavier_initializer()) 
    return memory
#memory:M*N*3
def choose_memory(partial_input,memory,use_type='c'):
    memory=tf.tile(tf.expand_dims(gt,axis=1),[BATCH_SIZE,1,1,1])
    model_num=memory.get_shape()[1].value
    dist=[]
    for i in range(model_num):
        if use_type=='c':
            dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(partial_input,memory[:,i])
            dist.append(tf.reduce_mean(tf.sqrt(dist1),axis=1,keepdims=True))
    dist=tf.concat(dist,axis=1)
    weights=dist/tf.reduce_sum(dist,axis=1,keepdims=True)#batch*M
    weights=tf.expand_dims(tf.expand_dims(weights,axis=-1),axis=-1)
    result=tf.reduce_sum(weights*memory,axis=1)
    return result
#memory:M*N*3
#gt:batch*N*3
def memory_loss(memory,gt,use_type='c'):
    memory=tf.tile(tf.expand_dims(gt,axis=1),[BATCH_SIZE,1,1,1])
    ptnum=memory.get_shape()[2].value
    model_num=memory.get_shape()[1].value
    gtdata=tf.tile(tf.expand_dims(gt,axis=1),[1,model_num,1,1])
    re_mem=tf.reshape(memory,[-1,ptnum,3])
    re_gt=tf.reshape(memory,[-1,ptnum,3])
    if use_type=='c':
        loss,_=chamfer_big(re_mem,re_gt)
    loss=tf.reshape(loss,[-1,model_num])
    match_loss=tf.reduce_min(loss,axis=1)#(batch,)
    rever_loss=tf.reduce_sum(loss,axis=1)-match_loss
    result=match_loss-0.1*rever_loss
    return result
def train_memory(sess,ops,train_writer,batch):
    ids,batch_point,npts,output_point=next(ops['train_gen'])
    feed_dict = {ops['pointcloud_pl']: batch_point,ops['gt_pl']:output_point,ops['type_pl']:'p'}
    _,loss=sess.run([ops['memstep'],ops['loss_mem']],feed_dict=feed_dict)
    
    if (batch+1) % 500 == 0:
        #result = sess.run(ops['merged'], feed_dict=feed_dict)
        #train_writer.add_summary(result, batch)
        print('epoch: %d'%(batch*BATCH_SIZE//ops['train_num']+1),'batch: %d' %batch)
        print('loss: ',loss)
#batch*1*1*128,batch*1*1*128
def init_decode_layer(scope,input_tensor,state_tensor,ptnum=128,mlp=[256,256],mlp1=[64],mlp11=[64],mlp2=[128,128],state_len=128,use_bnorm=False):
    with tf.variable_scope(scope):
        #input_info=conv2d('input_trans',input_tensor,256,[1,1],padding='VALID')
        if state_tensor is not None:
            input_info=conv2d('input_trans',input_tensor,128,[1,1],padding='VALID',use_bnorm=use_bnorm)
            state_info=conv2d('state_trans',state_tensor,128,[1,1],padding='VALID',use_bnorm=use_bnorm)
            new_state=tf.concat([input_info,state_info],axis=-1)
        else:
            new_state=conv2d('input_trans',input_tensor,256,[1,1],padding='VALID',use_bnorm=use_bnorm)
        for i,outchannel in enumerate(mlp):
            new_state=conv2d('basic_state%d'%i,new_state,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        points_out=new_state
        #for i,outchannel in enumerate(mlp1):
        #    points_out=conv2d('points%d'%i,points_out,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        points_out=conv2d('points_out',points_out,3*ptnum+12,[1,1],padding='VALID',activation_func=None)
        transmat,movemat=points_out[:,:,:,-12:-3],points_out[:,:,:,-3:]
        transmat,movemat=tf.reshape(transmat,[-1,3,3]),tf.reshape(movemat,[-1,1,3])
        points_out=points_out[:,:,:,:-12]
        points_out=tf.reshape(tf.nn.tanh(points_out),[-1,ptnum,3])
        points_out=tf.matmul(points_out,transmat)+movemat
        #points_out=tf.concat([tf.tile(input_info,[1,ptnum,1,1]),points_out],axis=-1)
        #for i,outchannel in enumerate(mlp11):
        #    points_out=conv2d('points2%d'%i,points_out,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        #points_out=tf.squeeze(conv2d('points2_out',points_out,3,[1,1],padding='VALID',activation_func=None),[2])

        state_out=conv2d('state_out',new_state,ptnum*16,[1,1],padding='VALID',use_bnorm=use_bnorm)
        state_out=tf.reshape(state_out,[-1,ptnum,1,16])
        state_out=tf.concat([state_out,tf.tile(new_state,[1,ptnum,1,1])],axis=-1)
        for i,outchannel in enumerate(mlp2):
            state_out=conv2d('state%d'%i,state_out,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        state_out=conv2d('state_outo',state_out,state_len,[1,1],padding='VALID',use_bnorm=use_bnorm)
        #batch*128*1*128,batch*128*1*3
    return points_out,state_out
##batch*1*1*128,batch*1*1*128
#def init_decode_layer(scope,input_tensor,state_tensor,ptnum=128,mlp=[256,256],mlp1=[64],mlp11=[64],mlp2=[128,128],state_len=128,use_bnorm=False):
#    with tf.variable_scope(scope):
#        #input_info=conv2d('input_trans',input_tensor,256,[1,1],padding='VALID')
#        if state_tensor is not None:
#            input_info=conv2d('input_trans',input_tensor,128,[1,1],padding='VALID',use_bnorm=use_bnorm)
#            state_info=conv2d('state_trans',state_tensor,128,[1,1],padding='VALID',use_bnorm=use_bnorm)
#            new_state=tf.concat([input_info,state_info],axis=-1)
#        else:
#            new_state=conv2d('input_trans',input_tensor,256,[1,1],padding='VALID',use_bnorm=use_bnorm)
#        for i,outchannel in enumerate(mlp):
#            new_state=conv2d('basic_state%d'%i,new_state,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
#        points_out=new_state
#        #for i,outchannel in enumerate(mlp1):
#        #    points_out=conv2d('points%d'%i,points_out,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
#        points_out=conv2d('points_out',points_out,3*ptnum,[1,1],padding='VALID',activation_func=None)
#        points_out=tf.reshape(points_out,[-1,ptnum,3])
#
#        #points_out=tf.concat([tf.tile(input_info,[1,ptnum,1,1]),points_out],axis=-1)
#        #for i,outchannel in enumerate(mlp11):
#        #    points_out=conv2d('points2%d'%i,points_out,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
#        #points_out=tf.squeeze(conv2d('points2_out',points_out,3,[1,1],padding='VALID',activation_func=None),[2])
#
#        state_out=conv2d('state_out',new_state,ptnum*16,[1,1],padding='VALID',use_bnorm=use_bnorm)
#        state_out=tf.reshape(state_out,[-1,ptnum,1,16])
#        state_out=tf.concat([state_out,tf.tile(new_state,[1,ptnum,1,1])],axis=-1)
#        for i,outchannel in enumerate(mlp2):
#            state_out=conv2d('state%d'%i,state_out,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
#        state_out=conv2d('state_outo',state_out,state_len,[1,1],padding='VALID',use_bnorm=use_bnorm)
#        #batch*128*1*128,batch*128*1*3
#    return points_out,state_out
def init_fn_decoder(scope,input_tensor,state_tensor,grid_length,grid_size=[8,8],mlp=[256,256],mlpfn0=[64,64],mlpfn1=[64,64],mlpfn2=[64,64],mlp2=[128],state_len=128):
    with tf.variable_scope(scope):
        xgrid_size=grid_size[0]
        ygrid_size=grid_size[1]
        xlength,ylength=grid_length
        if state_tensor is not None:
            input_info=conv2d('input_trans',input_tensor,128,[1,1],padding='VALID')
            state_info=conv2d('state_trans',state_tensor,128,[1,1],padding='VALID')
            new_state=tf.concat([input_info,state_info],axis=-1)
        else:
            new_state=conv2d('input_trans',input_tensor,256,[1,1],padding='VALID')
        for i,outchannel in enumerate(mlp):
            new_state=conv2d('basic_state%d'%i,new_state,outchannel,[1,1],padding='VALID')
        
        xgrid_feat=-xlength+2*xlength*tf.tile(tf.reshape(tf.linspace(0.0,1.0,xgrid_size),[1,1,-1,1]),[tf.shape(grid_feat)[0],tf.shape(grid_feat)[1],1,1])#batch*1*xgrid*1
        ygrid_feat=-ylength+2*ylength*tf.tile(tf.reshape(tf.linspace(0.0,1.0,ygrid_size),[1,1,-1,1]),[tf.shape(grid_feat)[0],tf.shape(grid_feat)[1],1,1])#batch*1*ygrid*1
        grid_feat=tf.concat([tf.tile(xgrid_feat,[1,1,ygrid_size,1]),tf.reshape(tf.tile(ygrid_feat,[1,1,1,xgrid_size]),[-1,ptnum,up_ratio,1])],axis=-1)#batch*1*up_ratio*2
        
        new_state=tf.concat([tf.tile(new_state,[1,1,up_ratio,1]),grid_feat],axis=-1)
        points_out=new_state
        for i,outchannel in enumerate(mlpfn0):
            points_out=conv2d('fn0_points%d'%i,points_out,outchannel,[1,1],padding='VALID')
        grid_feat=grid_feat+conv2d('fn0_points_out',points_out,2,[1,1],padding='VALID',activation_func=None)
        points_out=tf.concat([grid_feat,new_state],axis=-1)
        new_state=points_out
        for i,outchannel in enumerate(mlpfn1):
            points_out=conv2d('fn1_points%d'%i,points_out,outchannel,[1,1],padding='VALID')
        points_out=conv2d('fn1_points_out',points_out,3,[1,1],padding='VALID',activation_func=None)
        points_out=tf.concat([points_out,new_state],axis=-1)
        new_state=points_out
        for i,outchannel in enumerate(mlpfn2):
            points_out=conv2d('fn2_points%d'%i,points_out,outchannel,[1,1],padding='VALID')
        points_out=conv2d('fn2_points_out',points_out,3,[1,1],padding='VALID',activation_func=None)
        
        new_state=tf.concat([new_state,points_out],axis=-1)
        points_out=tf.reshape(points_out,[-1,ptnum*up_ratio,3])

        for i,outchannel in enumerate(mlp2):
            new_state=conv2d('state%d'%i,new_state,outchannel,[1,1],padding='VALID')
        state_out=conv2d('state_out%d'%i,new_state,state_len,[1,1],padding='VALID') 
        return points_out,state_out
def channel_shuffle(scope,x):
    with tf.variable_scope(scope):
        group_num=x.get_shape()[1].value
        feat_num=x.get_shape()[-1].value
        x_tr=tf.transpose(x,[0,3,2,1])
        new_x=tf.reshape(x_tr,[-1,group_num,1,feat_num])
    return new_x
def refine_layer(scope,ptcoor,feat,feat2,mlp=[128,64,64],mlp2=[128,128],mlpself=[128,128],reuse=False,no_feat=False,use_bnorm=False):
    with tf.variable_scope(scope,reuse=reuse):
        coor=tf.expand_dims(ptcoor,axis=2)
        featself=None
        ptnum=ptcoor.get_shape()[1].value
        #if mlpself is not None:
        #    featself=coor
        #    for i,outchannel in enumerate(mlp):
        #        featself=conv2d('self_layers%d'%i,featself,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        #    featself=tf.tile(tf.reduce_max(featself,axis=1,keepdims=True),[1,coor.get_shape()[1].value,1,1])
        tensor=coor
        if feat is not None:
            tensor=tf.concat([tensor,tf.tile(feat,[1,coor.get_shape()[1].value,1,1])],axis=-1)
        if featself is not None:
            tensor=tf.concat([tensor,featself],axis=-1)
        #if feat2 is not None:
        #    tensor=tf.concat([tensor,feat2],axis=-1)
        if feat2 is not None:
            statelen=feat2.get_shape()[-1].value
        else:
            statelen=feat.get_shape()[-1].value

        for i,outchannel in enumerate(mlpself):
            tensor=conv2d('ini_layer%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        tensorword=tensor
        maxtensor=tf.reduce_max(tensor,axis=1,keepdims=True)
        tensor=tf.concat([coor,tf.tile(maxtensor,[1,ptnum,1,1])],axis=-1)

        for i,outchannel in enumerate(mlp):
            tensor=conv2d('refine_layers%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        newvec=conv2d('refine_layer_final',tensor,3,[1,1],padding='VALID',activation_func=tf.nn.tanh)
        newvec=tf.squeeze(newvec,[2])

        newcoor=newvec+ptcoor
        tf.add_to_collection(scope+str(ptnum),newvec)
        if no_feat:
            newstate=None
        else:
            tensor=tf.concat([tf.expand_dims(newcoor,axis=2),feat2,tf.tile(feat,[1,feat2.get_shape()[1].value,1,1])],axis=-1)
            for i,outchannel in enumerate(mlp2):
                tensor=conv2d('feat_refine%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
            newfeat=conv2d('feat_refine_final',tensor,statelen,[1,1],padding='VALID',activation_func=tf.nn.tanh)
            newstate=newfeat+feat2
    return newcoor,newstate
#def refine_layer(scope,ptcoor,feat,feat2,up_ratio=16,mlp=[128,64,64],mlp2=[128,128],mlpself=[128,128],reuse=False,no_feat=False,use_bnorm=False):
#    with tf.variable_scope(scope,reuse=reuse):
#        coor=tf.expand_dims(ptcoor,axis=2)
#        featself=None
#        ptnum=ptcoor.get_shape()[1].value
#        statelen=feat2.get_shape()[-1].value
#        localnum=int(ptnum/up_ratio)
#        #if mlpself is not None:
#        #    featself=coor
#        #    for i,outchannel in enumerate(mlp):
#        #        featself=conv2d('self_layers%d'%i,featself,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
#        #    featself=tf.tile(tf.reduce_max(featself,axis=1,keepdims=True),[1,coor.get_shape()[1].value,1,1])
#        tensor=coor
#        tensor=tf.concat([tensor,feat2],axis=-1)
#        tensor=tf.reshape(tensor,[-1,localnum,up_ratio,statelen+3])
#        tensor=tf.concat([tensor,tf.tile(feat,[1,localnum,up_ratio,1])],axis=-1)
#        #if feat is not None:
#        #    tensor=tf.concat([tensor,tf.tile(feat,[1,coor.get_shape()[1].value,1,1])],axis=-1)
#        #if featself is not None:
#        #    tensor=tf.concat([tensor,featself],axis=-1)
#        #if feat2 is not None:
#        #    tensor=tf.concat([tensor,feat2],axis=-1)
#        #if feat2 is not None:
#        #    statelen=feat2.get_shape()[-1].value
#        #else:
#        #    statelen=feat.get_shape()[-1].value
#
#        for i,outchannel in enumerate(mlpself):
#            tensor=conv2d('ini_layer%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
#        tensorword=tensor
#        #localmax=tf.reduce_max(tensor,axis=2,keepdims=True)#batch*64*1*statelen
#        #localtensor=tf.reshape(tf.tile(localmax,[1,1,up_ratio,1]),[-1,ptnum,1,mlpself[-1]])
#        maxtensor=tf.reduce_max(tensor,axis=[1,2],keepdims=True)#batch*1*1*statelen
#        tensor=tf.concat([coor,tf.tile(maxtensor,[1,ptnum,1,1])],axis=-1)
#
#        for i,outchannel in enumerate(mlp):
#            tensor=conv2d('refine_layers%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
#        newvec=conv2d('refine_layer_final',tensor,3,[1,1],padding='VALID',activation_func=None)
#        newvec=tf.squeeze(newvec,[2])
#
#        newcoor=newvec+ptcoor
#        tf.add_to_collection(scope+str(ptnum),newvec)
#        if no_feat:
#            newstate=None
#        else:
#            #tensor=tf.concat([tf.expand_dims(newcoor,axis=2),feat2,tf.tile(feat,[1,feat2.get_shape()[1].value,1,1])],axis=-1)
#            tensor=tf.concat([feat2,tf.tile(maxtensor,[1,feat2.get_shape()[1].value,1,1])],axis=-1)
#            for i,outchannel in enumerate(mlp2):
#                tensor=conv2d('feat_refine%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
#            newfeat=conv2d('feat_refine_final',tensor,statelen,[1,1],padding='VALID',activation_func=None) 
#            newstate=newfeat+feat2
#    return newcoor,newstate
def res_layer(scope,ptcoor,mlp=[64,128,256],mlp2=[256,128,64]):
    with tf.variable_scope(scope):
        coor=tf.expand_dims(ptcoor,axis=2)
        tensor=coor
        for i,outchannel in enumerate(mlp):
            tensor=conv2d('refine_layers%d'%i,tensor,outchannel,[1,1],padding='VALID')
        tensor=tf.reduce_max(tensor,axis=1,keepdims=True)
        tensor=tf.concat([coor,tf.tile(tensor,[1,tf.shape(coor)[1],1,1])],axis=-1)
        for i,outchannel in enumerate(mlp):
            tensor=conv2d('res_layers%d'%i,tensor,outchannel,[1,1],padding='VALID')
        tensor=conv2d('res_layer_final',tensor,3,[1,1],padding='VALID',activation_func=None)
        coor=coor+tensor#batch*16384*1*3
    return tf.squeeze(coor,axis=[2])
#batch*1*1*128,batch*128*1*128
def decode_cell(scope,input_tensor,center,state_tensor,lastcode,up_ratio=4,up_channel=1,mlp=[256,256],mlp1=[64],mlp2=[128,128],mlp_mask=[128,128],mlp_expand=[128],state_len=128,reuse=False,use_bnorm=False):
    with tf.variable_scope(scope,reuse=reuse):
        ptnum=state_tensor.get_shape()[1].value
        tensor_length=input_tensor.get_shape()[-1].value
        #print(input_tensor,state_tensor)
        #cen_tensor=conv2d('cen2tensor',tf.expand_dims(center,axis=2),64,[1,1],padding='VALID',use_bnorm=use_bnorm)
        cen_tensor=tf.expand_dims(center,axis=2)
        mask_tensor=tf.concat([cen_tensor,tf.tile(input_tensor,[1,ptnum,1,1])],axis=-1)
        for i,outchannel in enumerate(mlp_mask):
            mask_tensor=conv2d('mlp_mask%d'%i,mask_tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        mask_tensor=conv2d('mask_tensor',mask_tensor,tensor_length,[1,1],padding='VALID',activation_func=tf.nn.relu)#batch*n*1*tensor_length
        
        #mask=tf.exp(mask_tensor)/tf.reduce_sum(tf.exp(mask_tensor),axis=-1,keepdims=True)
        input_info=mask_tensor*input_tensor
        #state_info=state_tensor
        
        #state_shuffle=channel_shuffle('channel_shuffle',state_info)
        input_info=conv2d('input_trans',input_info,256,[1,1],padding='VALID',use_bnorm=use_bnorm)
        state_info=conv2d('state_trans',state_tensor,128,[1,1],padding='VALID',use_bnorm=use_bnorm)
        if lastcode is not None:
            new_state=tf.concat([input_info,state_info,tf.tile(lastcode,[1,ptnum,1,1])],axis=-1)
        else:
            new_state=tf.concat([input_info,state_info],axis=-1)
        #new_state=tf.concat([input_info,state_info,state_shuffle],axis=-1)
        #new_state=tf.tile(input_info,[1,ptnum,1,1])*state_info
        newstatelist=[]

        #for i in range(up_ratio):
        #    newstate=new_state
        #    for j,outchannel in enumerate(mlp):
        #        newstate=conv2d('basic_state%d_%d'%(i,j),newstate,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        #    newstatelist.append(newstate)
        #new_state=tf.concat(newstatelist,axis=2)
        for i,outchannel in enumerate(mlp):
            new_state=conv2d('basic_state%d'%i,new_state,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        points_out=new_state#batch*64*1*256
        print('***************',points_out,new_state)
        for i,outchannel in enumerate(mlp1):
            points_out=conv2d('points%d'%i,points_out,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        #points_out=conv2d('points_out',points_out,3*up_ratio,[1,1],padding='VALID',activation_func=None)
        points_out=conv2d('points_out',points_out,3*up_ratio,[1,1],padding='VALID',activation_func=tf.nn.tanh)
        #points_out=conv2d('points_out',points_out,3,[1,1],padding='VALID',activation_func=None)
        #points_out=tf.reshape(points_out,[-1,ptnum,up_channel,up_ratio,3])
        #points_out=tf.reduce_sum(points_out,axis=2)
        #points_out=tf.tile(tf.expand_dims(center,axis=2),[1,1,up_ratio,1])+points_out
        #points_out=tf.reshape(points_out,[-1,ptnum*up_ratio,3])
        #ptfeats=tf.reshape(tf.tile(new_state,[1,1,up_ratio,1]),[-1,ptnum*up_ratio,1,mlp[-1]])
        #for level in range(up_level):
        #    if level<1:
        #        mreuse=False
        #    else:
        #        mreuse=True
        #    points_out=refine_layer('refine_level',points_out,feat=None,feat2=ptfeats,mlpself=None,reuse=mreuse)
 
        points_move=tf.reshape(points_out,[-1,ptnum,up_ratio,3])
        #tf.add_to_collection(scope+str(ptnum),tf.reshape(points_out,[-1,ptnum,up_ratio,3])-tf.tile(tf.expand_dims(center,axis=2),[1,1,up_ratio,1]))
        tf.add_to_collection(scope+str(ptnum),points_move)
        #print(center,points_out) 
        points_out=tf.tile(tf.expand_dims(center,axis=2),[1,1,up_ratio,1])+points_move
        points_out=tf.reshape(points_out,[-1,ptnum*up_ratio,3])
        #points_out=refine_layer(points_out,input_tensor)
        #points_out=tf.concat([tf.tile(input_info,[1,ptnum*up_ratio,1,1]),points_out],axis=-1)
        #for i,outchannel in enumerate(mlp11):
        #    points_out=conv2d('points2%d'%i,points_out,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        #points_out=tf.squeeze(conv2d('points2_out',points_out,3,[1,1],padding='VALID',activation_func=None),[2])
        #print(points_out)
        new_state=tf.concat([new_state,tf.tile(input_tensor,[1,ptnum,1,1])],axis=-1)
        for i,outchannel in enumerate(mlp2):
            new_state=conv2d('state%d'%i,new_state,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        #new_state=conv2d('state_out%d'%i,new_state,state_len,[1,1],padding='VALID',use_bnorm=use_bnorm,activation_func=None)
        #new_state=conv2d('state',new_state,state_len*up_ratio,[1,1],padding='VALID',use_bnorm=use_bnorm)
        #new_state=tf.concat([new_state,tf.tile(input_tensor,[1,ptnum,1,1])],axis=-1)
        newnew=new_state
        for i in range(up_ratio):
            for j,outchannel in enumerate(mlp_expand):
                newnew=conv2d('state_expand%d_%d'%(i,j),newnew,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
            newnew=conv2d('state_expand%d'%i,newnew,state_len,[1,1],padding='VALID',use_bnorm=use_bnorm,activation_func=tf.nn.leaky_relu)
            if i<1:
                new_state=newnew 
            else:
                new_state=tf.concat([new_state,newnew],axis=2)
        state_move=new_state
        new_state=tf.tile(state_tensor,[1,1,up_ratio,1])+state_move
        new_state=tf.reshape(new_state,[-1,up_ratio*ptnum,1,state_len])
    return points_out,new_state#,points_move,state_move
def fn_decode_cell(scope,input_tensor,center,state_tensor,lastcode=None,up_ratio=16,mlp=[256,256],mlp_trans=[128,64],mlp_transmat=[64,9],mlp_grid=[128,64,3],\
                   mlp_mask=[128,128],mlpfn0=[64,64],mlpfn1=[64,64],mlpfn2=[64,64],mlp2=[128,128],grid_scale=0.05,state_len=128,reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        bsize=tf.shape(input_tensor)[0]
        ptnum=state_tensor.get_shape()[1].value
        digit=tf.log(up_ratio*1.0)/tf.log(2.0)

        tensor_length=input_tensor.get_shape()[-1].value
        if center is None:
            cen_tensor=tf.zeros([tf.shape(input_tensor)[0],1,1,3])
        else:
            cen_tensor=tf.expand_dims(center,axis=2)
        mask_tensor=tf.concat([cen_tensor,tf.tile(input_tensor,[1,ptnum,1,1])],axis=-1)
        for i,outchannel in enumerate(mlp_mask):
            mask_tensor=conv2d('mlp_mask%d'%i,mask_tensor,outchannel,[1,1],padding='VALID')
        mask_tensor=conv2d('mask_tensor',mask_tensor,tensor_length,[1,1],padding='VALID',activation_func=tf.nn.relu)#batch*n*1*tensor_length
        input_info=mask_tensor*input_tensor

        input_info=conv2d('input_trans',input_info,256,[1,1],padding='VALID')
        state_info=conv2d('state_trans',state_tensor,128,[1,1],padding='VALID')
        if lastcode is not None:
            new_state=tf.concat([input_info,state_info,tf.tile(lastcode,[1,ptnum,1,1])],axis=-1)
        else:
            new_state=tf.concat([input_info,state_info],axis=-1)
        #new_state=tf.concat([new_state,tf.expand_dims(grid_feat,axis=2)],axis=-1)
        for i,outchannel in enumerate(mlp):
            new_state=conv2d('basic_state%d'%i,new_state,outchannel,[1,1],padding='VALID')

        grid_feat=new_state
        for i,outchannel in enumerate(mlp_grid[:-1]):
            grid_feat=conv2d('basic_grid%d'%i,grid_feat,outchannel,[1,1],padding='VALID')#batch*64*1*11
        raw_grid_feat=conv2d('basic_grid-111',grid_feat,mlp_grid[-1],[1,1],padding='VALID',activation_func=None)#batch*64*1*11
        #raw_grid_feat=conv2d('basic_grid-1%d'%i,grid_feat,2*grid_size**2,[1,1],padding='VALID',activation_func=None)
        #grid_feat=tf.reshape(raw_grid_feat,[-1,ptnum,grid_size*grid_size,2])
        #dev_feat=grid_feat
        #for i,outchannel in enumerate(mlp_devide):
        #    dev_feat=conv2d('tensor_devided%d'%i,dev_feat,outchannel,[1,1],padding='VALID')#batch*64*1*128
        #dev_feat=conv2d('feat_devided%d'%i,dev_feat,mlp[-1],[1,1],padding='VALID')*tf.tile(new_state,[1,1,up_ratio,1])
        #trans_matrix=tf.reshape(raw_grid_feat[:,:,:,2:],[-1,ptnum,1,3,3])
        #initpts=tf.reshape(raw_grid_feat[:,:,:,11:],[-1,ptnum,up_ratio,2])
        #cenpts=(tf.reduce_max(initpts,axis=-2,keepdims=True)+tf.reduce_min(initpts,axis=-2,keepdims=True))/2
        #initpts=2*(initpts-cenpts)/(tf.reduce_max(initpts,axis=-2,keepdims=True)-tf.reduce_min(initpts,axis=-2,keepdims=True))
        xylength=tf.square(raw_grid_feat[:,:,:,:2])
        grid_size=int(sqrt(up_ratio))
        grids=tf.random_uniform((bsize,ptnum,grid_size, 2), minval=-1,maxval=1, dtype=tf.float32)
        xgrids=tf.tile(tf.expand_dims(grids[:,:,:,:1],axis=3),[1,1,1,grid_size,1])
        ygrids=tf.tile(tf.expand_dims(grids[:,:,:,1:],axis=2),[1,1,grid_size,1,1])
        grid_feat=tf.reshape(tf.concat([xgrids,ygrids],axis=-1),[-1,ptnum,up_ratio,2])

        grid_feat=2*grid_feat*xylength-xylength#batch*64*up_ratio*2,-1~1 grids

        raw_state=new_state
        new_state=tf.concat([tf.tile(new_state,[1,1,up_ratio,1]),grid_feat],axis=-1)
        points_out=new_state
        #for i,outchannel in enumerate(mlpfn0):
        #    points_out=conv2d('fn0_points%d'%i,points_out,outchannel,[1,1],padding='VALID')
        #grid_feat=grid_feat+conv2d('fn0_points_out',points_out,2,[1,1],padding='VALID',activation_func=None)

        #z_feat=tf.zeros([tf.shape(grid_feat)[0],tf.shape(grid_feat)[1],tf.shape(grid_feat)[2],1])
        #grid_feat=tf.concat([grid_feat,z_feat],axis=-1)#batch*64*16*3*1
        #grid_feat=tf.matmul(tf.tile(trans_matrix,[1,1,up_ratio,1,1]),tf.expand_dims(grid_feat,axis=-1))
        #grid_feat=tf.squeeze(grid_feat,axis=-1)

        #new_state=tf.concat([tf.tile(new_state,[1,1,up_ratio,1]),grid_feat],axis=-1)
        points_out=tf.concat([grid_feat,new_state],axis=-1)
        #new_state=points_out
        #points_out=new_state
        for i,outchannel in enumerate(mlpfn1):
            points_out=conv2d('fn1_points%d'%i,points_out,outchannel,[1,1],padding='VALID')
        points_out=conv2d('fn1_points_out',points_out,3,[1,1],padding='VALID',activation_func=None)

        #points_out=tf.matmul(tf.tile(trans_matrix,[1,1,up_ratio,1,1]),tf.expand_dims(points_out,axis=-1))
        #points_out=tf.squeeze(points_out,axis=-1)

        #points_out=tf.concat([points_out,new_state],axis=-1)
        #new_state=points_out
        #for i,outchannel in enumerate(mlpfn2):
        #    points_out=conv2d('fn2_points%d'%i,points_out,outchannel,[1,1],padding='VALID')
        #points_out=conv2d('fn2_points_out',points_out,3,[1,1],padding='VALID',activation_func=None)

        #tf.add_to_collection(scope+'_fold_grid'+str(ptnum),points_out-grid_feat)
        #new_state=tf.concat([new_state,points_out],axis=-1)

        #trans_state=raw_state
        #for i,outchannel in enumerate(mlp_trans):
        #    trans_state=conv2d('trans_local%d'%i,trans_state,outchannel,[1,1],padding='VALID')
        #trans_state=tf.reduce_max(trans_state,axis=2,keepdims=True)
        #for i,outchannel in enumerate(mlp_transmat):
        #    trans_state=conv2d('trans_feat%d'%i,trans_state,outchannel,[1,1],padding='VALID')
        #trans_matrix=conv2d('transmat',trans_state,9,[1,1],padding='VALID',activation_func=None)
        #trans_matrix=tf.reshape(trans_matrix,[-1,ptnum,1,3,3])
        ##z_feat=tf.ones([tf.shape(points_out)[0],tf.shape(points_out)[1],tf.shape(points_out)[2],1])
        ##points_out=tf.expand_dims(tf.concat([points_out,z_feat],axis=-1),axis=-1)#batch*64*16*3*1

        #points_out=tf.matmul(tf.tile(trans_matrix,[1,1,up_ratio,1,1]),tf.expand_dims(points_out,axis=-1))
        #points_out=tf.squeeze(points_out,axis=-1)
        new_state=tf.concat([new_state,points_out],axis=-1)

        tf.add_to_collection(scope+str(ptnum),points_out)
        move=points_out
        points_out=tf.tile(cen_tensor,[1,1,up_ratio,1])+move
        points_out=tf.reshape(points_out,[-1,ptnum*up_ratio,3])

        for i,outchannel in enumerate(mlp2):
            new_state=conv2d('state%d'%i,new_state,outchannel,[1,1],padding='VALID')
        new_state=conv2d('state_out%d'%i,new_state,state_len,[1,1],padding='VALID')
        new_state=conv2d('fn_state_out',new_state,state_len,[1,1],padding='VALID',activation_func=None)
        new_state=new_state+tf.tile(state_tensor,[1,1,up_ratio,1])
        new_state=tf.reshape(new_state,[-1,ptnum*up_ratio,1,state_len])

    return points_out,new_state
def cut_decode_cell(scope,input_tensor,center,state_tensor,lastcode=None,up_ratio=16,mlp=[256,256],mlp_trans=[128,64],mlp_transmat=[64,9],mlp_grid=[128,64,21],\
                   mlp_mask=[128,128],mlpfn0=[64,32],mlpfn1=[64,32],mlpfn2=[64,32],mlp2=[128,128],grid_scale=0.05,state_len=128,reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        ptnum=state_tensor.get_shape()[1].value
        layernum=4
        grid_size=int(sqrt(up_ratio/layernum))
        #digit=tf.log(up_ratio*1.0/4)/tf.log(2.0)
        #digit=2
        tensor_length=input_tensor.get_shape()[-1].value
        if center is None:
            cen_tensor=tf.zeros([tf.shape(input_tensor)[0],1,1,3])
        else:
            cen_tensor=tf.expand_dims(center,axis=2)
        mask_tensor=tf.concat([cen_tensor,tf.tile(input_tensor,[1,ptnum,1,1])],axis=-1)
        for i,outchannel in enumerate(mlp_mask):
            mask_tensor=conv2d('mlp_mask%d'%i,mask_tensor,outchannel,[1,1],padding='VALID')
        mask_tensor=conv2d('mask_tensor',mask_tensor,tensor_length,[1,1],padding='VALID',activation_func=tf.nn.relu)#batch*n*1*tensor_length 
        input_info=mask_tensor*input_tensor
        
        input_info=conv2d('input_trans',input_info,256,[1,1],padding='VALID')
        state_info=conv2d('state_trans',state_tensor,128,[1,1],padding='VALID')
        if lastcode is not None:
            new_state=tf.concat([input_info,state_info,tf.tile(lastcode,[1,ptnum,1,1])],axis=-1)
        else:
            new_state=tf.concat([input_info,state_info],axis=-1)
        for i,outchannel in enumerate(mlp):
            new_state=conv2d('basic_state%d'%i,new_state,outchannel,[1,1],padding='VALID')

        grid_feat=new_state
        for i,outchannel in enumerate(mlp_grid[:-1]):
            grid_feat=conv2d('basic_grid%d'%i,grid_feat,outchannel,[1,1],padding='VALID')#batch*64*1*11
        raw_grid_feat=conv2d('basic_grid-111',grid_feat,mlp_grid[-1],[1,1],padding='VALID',activation_func=None)#batch*64*1*11
        xylength=tf.nn.relu(raw_grid_feat[:,:,:,:8])#batch*64*1*8
        height=tf.expand_dims(raw_grid_feat[:,:,:,8:12],axis=-1)#batch*64*1*4
        trans_matrix=tf.reshape(raw_grid_feat[:,:,:,12:],[-1,ptnum,1,3,3])#batch*64*1*3*3
        #layer_trans=tf.reshape(raw_grid_feat[:,:,:,21:57],[-1,ptnum,layernum,3,3])#batch*64*4*3*3

        #ratios=tf.squeeze(tf.nn.relu(raw_grid_feat[:,:,:,-1]),axis=-1)#batch*64
        #ratios=4*ptnum*tf.exp(ratios)/tf.reduce_sum(tf.exp(ratios),axis=1,keepdims=True)
        #ratios=ratios[:,:-1]//1.0
        #ratios=tf.concat([ratios,4*ptnum-tf.reduce_sum(ratios,axis=1,keepdims=True)],axis=-1)
        #layernum=tf.reshape(ratios,[-1,ptnum,1,1])

        grid_feat=-1+2*tf.tile(tf.reshape(tf.linspace(0.0,1.0,grid_size),[1,1,-1,1]),[tf.shape(raw_grid_feat)[0],ptnum,1,2])#batch*64*2*2
        #zgrid_feat=-1+2*tf.tile(tf.reshape(tf.linspace(0.0,1.0,layernum),[1,1,-1,1]),[tf.shape(raw_grid_feat)[0],ptnum,1,1])#batch*64*4*1
        
        xgrid_feat=tf.expand_dims(grid_feat[:,:,:,0],axis=-1)
        ygrid_feat=tf.expand_dims(grid_feat[:,:,:,1],axis=-1)
        grid_feat=tf.concat([tf.tile(xgrid_feat,[1,1,grid_size,1]),tf.reshape(tf.tile(ygrid_feat,[1,1,1,grid_size]),[-1,ptnum,grid_size*grid_size,1])],axis=-1)#batch*64*4*2

        xylength=tf.reshape(xylength,[-1,ptnum,int(up_ratio/layernum),2])
        grid_feat=grid_feat*xylength#batch*64*4*2,-1~1 grids
        grid_feat=tf.reshape(grid_feat,[-1,ptnum,4,2])#batch*64*4*2
        grid_feat=tf.tile(grid_feat,[1,1,layernum,1])
        height=tf.reshape(height,[-1,ptnum,layernum,1])
        zgrid_feat=tf.reshape(tf.tile(height,[1,1,1,int(up_ratio/layernum)]),[-1,ptnum,up_ratio,1])#batch*64*16*1
        grid_feat=tf.concat([grid_feat,zgrid_feat],axis=-1)#batch*64*16*3
        
        raw_state=new_state
        new_state=tf.concat([tf.tile(new_state,[1,1,up_ratio,1]),grid_feat],axis=-1)
        points_out=new_state
        #for i,outchannel in enumerate(mlpfn0):
        #    points_out=conv2d('fn0_points%d'%i,points_out,outchannel,[1,1],padding='VALID')
        #grid_feat=grid_feat+conv2d('fn0_points_out',points_out,2,[1,1],padding='VALID',activation_func=None)

        #z_feat=tf.zeros([tf.shape(grid_feat)[0],tf.shape(grid_feat)[1],tf.shape(grid_feat)[2],1])
        #grid_feat=tf.concat([grid_feat,z_feat],axis=-1)#batch*64*16*3*1
        #grid_feat=tf.matmul(tf.tile(trans_matrix,[1,1,up_ratio,1,1]),tf.expand_dims(grid_feat,axis=-1))
        #grid_feat=tf.squeeze(grid_feat,axis=-1)
        
        #new_state=tf.concat([tf.tile(new_state,[1,1,up_ratio,1]),grid_feat],axis=-1)
        #points_out=tf.concat([grid_feat,new_state],axis=-1)
        #new_state=points_out
        #points_out=new_state
        for i,outchannel in enumerate(mlpfn1):
            points_out=conv2d('fn1_points%d'%i,points_out,outchannel,[1,1],padding='VALID')
        points_out=conv2d('fn1_points_out',points_out,3,[1,1],padding='VALID',activation_func=None)
        points_out1=points_out+grid_feat
        #points_out=tf.matmul(tf.tile(trans_matrix,[1,1,up_ratio,1,1]),tf.expand_dims(points_out,axis=-1))
        #points_out=tf.squeeze(points_out,axis=-1)
        points_out=tf.concat([points_out1,tf.tile(raw_state,[1,1,up_ratio,1])],axis=-1)
        new_state=points_out
        for i,outchannel in enumerate(mlpfn2):
            points_out=conv2d('fn2_points%d'%i,points_out,outchannel,[1,1],padding='VALID')
        points_out=conv2d('fn2_points_out',points_out,3,[1,1],padding='VALID',activation_func=None)
        points_out=points_out+points_out1
        #points_out=tf.expand_dims(tf.concat([points_out,z_feat],axis=-1),axis=-1)#batch*64*16*3*1

        points_out=tf.matmul(tf.tile(trans_matrix,[1,1,up_ratio,1,1]),tf.expand_dims(points_out,axis=-1)) 
        points_out=tf.squeeze(points_out,axis=-1)
        new_state=tf.concat([new_state,points_out],axis=-1)

        tf.add_to_collection(scope+str(ptnum),points_out)
        points_out=tf.tile(cen_tensor,[1,1,up_ratio,1])+points_out
        points_out=tf.reshape(points_out,[-1,ptnum*up_ratio,3])

        #newnew=raw_state
        #for i,outchannel in enumerate(mlp2):
        #    newnew=conv2d('state%d'%i,newnew,outchannel,[1,1],padding='VALID')
        #newnew=conv2d('state_out%d'%i,newnew,up_ratio*state_len,[1,1],padding='VALID')
        #newnew=tf.reshape(newnew,[-1,ptnum*up_ratio,1,state_len])
        for i,outchannel in enumerate(mlp2):
            new_state=conv2d('state%d'%i,new_state,outchannel,[1,1],padding='VALID')
        new_state=conv2d('state_out%d'%i,new_state,state_len,[1,1],padding='VALID')
        new_state=tf.reshape(new_state,[-1,ptnum*up_ratio,1,state_len])
        
    return points_out,new_state

#state_tensor:batch*1*1*128
#code_tensor:batch*1*1*128
def re_decoder(state_tensor,rnum=3):
    with tf.variable_scope('D'):
        #init_grid,init_size,multi1,multi2=multiple_calculate('digit_distribute',[tf.get_collection('code1')[0],tf.get_collection('code2')[0],tf.get_collection('code3')[0]],mlp=[128,128,64,5],ptnum=16384)
        #points1,state=init_fn_decoder('init_cell',tf.get_collection('code1')[0],None,grid_length=init_size,grid_size=init_grid,mlp=[256,256],mlpfn0=[64,64],mlpfn1=[64,64],mlpfn2=[64,64],mlp2=[128,128],state_len=128) 
        #points1,state=init_decode_layer('init_cell',tf.concat([tf.get_collection('code1')[0],tf.get_collection('code2')[0],tf.get_collection('code3')[0]],axis=-1),None,ptnum=64,mlp1=[128],mlp2=[256,256],state_len=128)
        points1,state=init_decode_layer('init_cell',tf.get_collection('code1')[0],None,ptnum=64,mlp1=[128],mlp2=[256,256],state_len=128)
        #points1,state=fn_decode_cell('decode_cell',tf.get_collection('code1')[0],None,tf.get_collection('code1')[0],grid_size=8)
        tf.add_to_collection('o128', points1)
        #points2,state=fn_decode_cell('decode_cell',tf.get_collection('code2')[0],points1,state,grid_size=[2,2],grid_scale=0.05)
        #points3,state=fn_decode_cell('decode_cell1',tf.get_collection('code3')[0],points2,state,grid_size=[2,2],grid_scale=0.02)
        points1,state=refine_layer('refine_layer',points1,feat=tf.concat([tf.get_collection('code1')[0],tf.get_collection('code2')[0]],axis=-1),feat2=state,mlpself=None)
        tf.add_to_collection('cen1', points1) 
        #points1=refine_layer('refine_layer',points1,feat=tf.get_collection('code1')[0],feat2=None,mlpself=None)
        #points1,state=fn_decode_cell('fn_decode_cell',tf.get_collection('code1')[0],None,state_tensor,grid_size=[32,32],grid_scale=0.5)
        #points2,state=decode_cell('decode_cell',tf.get_collection('code2')[0],points1,state,tf.get_collection('code1')[0],up_ratio=16,mlp1=[64,32],mlp2=[256,256],state_len=128)
        points2,state=fn_decode_cell('decode_cell',tf.get_collection('code2')[0],points1,state,up_ratio=16,state_len=128,reuse=False)
        tf.add_to_collection('o512', points2)
        points2,state=refine_layer('refine_layer',points2,feat=tf.concat([tf.get_collection('code2')[0],tf.get_collection('code3')[0]],axis=-1),feat2=state,mlpself=None,reuse=True)
        tf.add_to_collection('cen2', points2)
        #points2=refine_layer('refine_layer',points2,feat=tf.get_collection('code2')[0],feat2=None,mlpself=None,reuse=True)
        #points3,state=decode_cell('decode_cell',tf.get_collection('code3')[0],points2,state,tf.get_collection('code2')[0],up_ratio=16,mlp1=[64,32],mlp2=[256,256],reuse=True,state_len=128)
        points3,state=fn_decode_cell('decode_cell',tf.get_collection('code3')[0],points2,state,up_ratio=16,state_len=128,reuse=True)
        #points3,state=refine_layer('refine_layer',points3,feat=tf.concat([tf.get_collection('code2')[0],tf.get_collection('code3')[0]],axis=-1),feat2=state,mlpself=None,reuse=True)
        #tf.add_to_collection('o128', points1)
        #tf.add_to_collection('o512', points2)
        tf.add_to_collection('o2048', points3)
    return points1,points2,points3
##state_tensor:batch*1*1*128
#code_tensor:batch*1*1*128
#target:batch*16384*3
def re_decoder_norefine(state_tensor,target,rnum=3):
    with tf.variable_scope('D'):
         ptnum=64
         _,points_tar2=sampling(ptnum*16,target,use_type='f')
         _,points_tar1=sampling(ptnum,points_tar2,use_type='f')
         #points_tar2=tf.expand_dims(points_tar2,axis=2)
         #points_tar1=tf.expand_dims(points_tar1,axis=2) 

         points1,state1=init_decode_layer('init_cell',tf.get_collection('code1')[0],None,ptnum=64,mlp1=[128],mlp2=[128,128],state_len=128)
         points2,state2=fn_decode_cell('decode_cell',tf.get_collection('code2')[0],points1,state1,up_ratio=16,reuse=False)
         points3,_=fn_decode_cell('decode_cell',tf.get_collection('code3')[0],points2,state2,up_ratio=16,reuse=True)
         tf.add_to_collection('o128', points1)
         tf.add_to_collection('o512', points2)
         tf.add_to_collection('o2048', points3)
         print(points_tar1)
         points_tar1,state1=refine_layer('refine_layer',points_tar1,feat=tf.concat([tf.get_collection('code1')[0],tf.get_collection('code2')[0]],axis=-1),feat2=state1,mlpself=None)
         points_tar2,state2=refine_layer('refine_layer',points_tar2,feat=tf.concat([tf.get_collection('code2')[0],tf.get_collection('code3')[0]],axis=-1),feat2=state2,mlpself=None,reuse=True)
         target_tar2,_=fn_decode_cell('decode_cell',tf.get_collection('code2')[0],points_tar1,state1,up_ratio=16,reuse=True)
         target_tar3,_=fn_decode_cell('decode_cell',tf.get_collection('code3')[0],points_tar2,state2,up_ratio=16,reuse=True)
         tf.add_to_collection('pts_tar1', points_tar1)
         tf.add_to_collection('pts_tar2', points_tar2)
         tf.add_to_collection('tar_tar2', target_tar2)
         tf.add_to_collection('tar_tar3', target_tar3)
    return points1,points2,points3
         
def fn_re_decoder(state_tensor,rnum=4):
    with tf.variable_scope('D'):
        points1,state=fn_decode_cell('decode_cell',tf.get_collection('code1')[0],None,state_tensor,grid_size=[16,16],grid_scale=0.5)
        points2,state=fn_decode_cell('decode_cell',tf.get_collection('code2')[0],points1,state,grid_size=[2,2],grid_scale=0.2,reuse=True)
        points3,state=fn_decode_cell('decode_cell',tf.get_collection('code3')[0],points2,state,grid_size=[2,2],grid_scale=0.05,reuse=True)
        points4,state=fn_decode_cell('decode_cell',tf.get_collection('code4')[0],points3,state,grid_size=[2,2],grid_scale=0.02,reuse=True)
        tf.add_to_collection('o256', points1)
        tf.add_to_collection('o1024', points2)
        tf.add_to_collection('o4096', points3)
        tf.add_to_collection('o16384',points4)
    return points1,points2,points3,points4
def normalize(tensor_data):
    tensormean=tf.reduce_mean(tensor_data,axis=0)
    tensorvar=tf.clip_by_value(tf.sqrt(tf.reduce_mean(tf.square(tensor_data-tensormean),axis=0)),1e-10,10)
    tensornorm=(tensor_data-tensormean)/tensorvar
    print(tensornorm)
    return tensornorm
def decompress_layer(scope,tensor,mlp,mlp1,mlp2,up_ratio=4):
    with tf.variable_scope(scope):
        new_point=tensor
        new_point_list=[]
        for i in range(up_ratio):
            newpoint=tensor
            for j, num_out_channel0 in enumerate(mlp):
                newpoint = conv2d('convall%d_%d'%(i,j),newpoint,num_out_channel0,[1,1])
            new_point_list.append(newpoint)
        new_point=tf.concat(new_point_list,axis=1)
        new_feat=new_point
        for j, num_out_channel in enumerate(mlp1):
            new_feat = conv2d('convfeat_%d'%j,new_feat, num_out_channel, [1,1]) 
        for k, num_out_channel2 in enumerate(mlp2):
            new_point = conv2d('convpt_%d'%k,new_point,num_out_channel2,[1,1])
        new_point = conv2d('convptout',new_point,3,[1,1],activation_func=None)
        return new_point,new_feat
def decom_ful(scope,tensor,mlp,mlp1,mlp2,up_ratio=4,fea_dim=128):
    with tf.variable_scope(scope):
        rawpt_len=tensor.get_shape()[1].value
        new_point=tensor
        for i, num_out_channel0 in enumerate(mlp):
            new_point = conv2d('ful_ini%d'%i,new_point,num_out_channel0,[1,1])
        new_feat=new_point
        for j, num_out_channel in enumerate(mlp1):
            new_feat = conv2d('fulfeat_%d'%j,new_feat, num_out_channel, [1,1])
        new_feat = conv2d('fulfeatout',new_feat,fea_dim*up_ratio,[1,1])
        new_feat = tf.reshape(new_feat,[-1,rawpt_len*up_ratio,1,fea_dim]) 
        for k, num_out_channel2 in enumerate(mlp2):
            new_point = conv2d('fulpt_%d'%k,new_point,num_out_channel2,[1,1])
        new_point = conv2d('convptout',new_point,3*up_ratio,[1,1],activation_func=None)
        new_point=tf.reshape(new_point,[-1,rawpt_len*up_ratio,1,3])
        return new_point,new_feat

def gradual_final1(tensor):
    tensor=tf.cast(tensor,tf.float32)
    featLen = tensor.get_shape()[-1].value
    batchNum = tf.shape(tensor)[0]
    input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])
    with tf.variable_scope('D128'):
        #net = conv2d('fully_layer1', input_tensor, 512, [1, 1])
        #net = conv2d('fully_layer2', net,512,[1,1])
        #net = conv2d('fully_layer3', net,384,[1,1],activation_func=None)
        #net128 = tf.reshape(net,[BATCH_SIZE,128,1,3])
        net128,netfeat128=decom_ful('decom_layer0',input_tensor,[512,256],[256],[256],128,128)
    tf.add_to_collection('o128', net128)
    with tf.variable_scope('D512'):
        tensor128=tf.concat([net128,netfeat128,tf.tile(input_tensor,multiples=[1,128,1,1])],axis=-1)
        net512,netfeat512=decom_ful('decom_layer1',tensor128,[512,256],[128,128],[128,64])         
    tf.add_to_collection('o512',net512)
    with tf.variable_scope('D2048'):
        tensor512=tf.concat([net512,netfeat512,tf.tile(input_tensor,multiples=[1,512,1,1])],axis=-1)
        net2048,netfeat=decom_ful('decom_layer2',tensor512,[512,256],[128,128],[128,64])
    tf.add_to_collection('o2048', net2048) 
    return tf.reshape(net128, [-1, 128, 3]),tf.reshape(net512, [-1, 512, 3]),tf.reshape(net2048, [-1, 2048, 3])


def gradual_final(tensor,ex_type='d'):
    #tensor=tf.cast(tensor,tf.float32)
    tensor=tf.cast(tensor,tf.float32)
    featLen = tensor.get_shape()[-1].value
    batchNum = tf.shape(tensor)[0]
    input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])
    with tf.variable_scope('D128'):
        net = deconv('deconv_layer1', inputs=input_tensor, output_shape=[batchNum, 128, 1, 128], kernel_size=[128, 1],stride=[1, 1], padding='VALID')
        #net = deconv('deconv_layer2', inputs=net, output_shape=[batchNum, 32, 8, 128], kernel_size=[16, 1], stride=[4, 1])

        netfeat128 = deconv('deconv_layer3', inputs=net, output_shape=[batchNum, 128, 8, 64], kernel_size=[1,8], stride=[1, 1],padding='VALID')
        net = conv2d('conv2d_layer4', netfeat128,64,[1,8],padding='VALID')

        net=tf.concat([net,tf.tile(input_tensor,multiples=[1,128,1,1])],axis=-1)
        net=conv2d('conv2d_layer5',net,64,[1,1])
        net128=conv2d('o128', net,3,[1,1],activation_func=None)
    tf.add_to_collection('o128', net128)
    with tf.variable_scope('D512'):
        #net=tf.concat([tf.reshape(netfeat128,[BATCH_SIZE,128,1,-1]),tf.tile(input_tensor,multiples=[1,128,1,1])],axis=-1)
        #net=tf.reshape(netfeat128,[BATCH_SIZE,128,1,-1])
        #net=conv2d('conv_layer',net,512,[1,1])
        net=tf.concat([net128,tf.reshape(netfeat128,[BATCH_SIZE,128,1,-1])],axis=-1)

        _,netl=local_net(scope='layer',sample_dim=128,xyz=tf.reshape(net128,[-1,128,3]),featvec=tf.squeeze(net),r_list=[0.2],k_list=[8],layers_list=[[128,128]],use_all=True)
        netl=tf.expand_dims(netl,axis=2)
        net=tf.concat([net,netl],axis=-1)
        
        net = deconv('deconv_layer7', inputs=net, output_shape=[batchNum, 128, 8, 128], kernel_size=[1, 8], stride=[1, 1],padding='VALID')
        netfeat512 = deconv('deconv_layer8', inputs=net, output_shape=[batchNum, 512, 8, 64], kernel_size=[4, 1], stride=[4, 1],padding='VALID')

        #netfeat512 = conv2d('layer5_conv2d' , net,64,[1,8],padding='VALID')
        #netfeat512 = deconv('deconv_layer5', inputs=net, output_shape=[batchNum, 512, 8, 128], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
        net=conv2d('conv2d_layer9',netfeat512,64,[1,8],padding='VALID')
        net=tf.concat([net,tf.tile(input_tensor,multiples=[1,512,1,1])],axis=-1)

        net=conv2d('conv2d_layer10',net,64,[1,1])
        net512 = conv2d('conv2d_layer11',net,3,[1,1],activation_func=None)
    tf.add_to_collection('o512',net512)
    with tf.variable_scope('D2048'):
        #net = tf.concat([tf.reshape(netfeat512,[BATCH_SIZE,512,1,-1]),tf.tile(input_tensor, multiples=[1, 512, 1, 1])], axis=-1)
        #net=conv2d('conv_layer',net,512,[1,1])
        net=tf.concat([net512,tf.reshape(netfeat512,[BATCH_SIZE,512,1,-1])],axis=-1)
        _,netl=local_net(scope='layer',sample_dim=512,xyz=tf.reshape(net512,[-1,512,3]),featvec=tf.squeeze(net),r_list=[0.4],k_list=[32],layers_list=[[128,128]],use_all=True)
        netl=tf.expand_dims(netl,axis=2)
        net=tf.concat([net,netl],axis=-1)
        #net = conv2d('conv2d_layer8',net,128,[1,1])
        net = deconv('deconv_layer12', inputs=net, output_shape=[batchNum,512 , 8, 128], kernel_size=[1, 8], stride=[1, 1],padding='VALID')
        net = deconv('deconv_layer13', inputs=net, output_shape=[batchNum,2048 , 8, 64], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
        #net = deconv('deconv_layer7', inputs=net, output_shape=[batchNum,2048 , 16, 128], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
        net=conv2d('conv2d_layer14',net,64,[1,8],padding='VALID')
        net=tf.concat([net,tf.tile(input_tensor,multiples=[1,2048,1,1])],axis=-1)
        net=conv2d('conv2d_layer15',net,64,[1,1])
        net2048 = conv2d('conv2d_layer16', net, 3, [1, 1],activation_func=None)
    tf.add_to_collection('o2048', net2048) 
    return tf.reshape(net128, [-1, 128, 3]),tf.reshape(net512, [-1, 512, 3]),tf.reshape(net2048, [-1, 2048, 3])


#def fully_initial(tensor,npoint,featdim,mlp,mlp1,mlp2):
     
def rbf_expand(xyz,input_tensor,mlp,k=16,n=1,proj_type='g',up_ratio=4,use_all=True):
    #xyz:batch*128*3
    #input_tensor:batch*128*1*128
    input_len=input_tensor.get_shape()[1].value
    input_tensor=tf.squeeze(input_tensor,[2])#batch*128*128
    dis_xyz=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(xyz,axis=2)-tf.expand_dims(xyz,axis=1)),axis=-1))#batch*128*128
    if use_all:
        base_xyz=tf.tile(tf.expand_dims(xyz,axis=2),[1,1,input_len,1]) #batch*128*128*3
        base_feat=tf.tile(tf.expand_dims(input_tensor,axis=2),[1,1,input_len,1])#batch*128*128*128
    else:
        #dis_xyz=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(xyz,axis=2)-tf.expand_dims(xyz,axis=1)),axis=-1))
        base_xyz,base_idx=get_kneighbor(input_tensor,dis_xyz,k)#batch*128*k*3
        base_feat=tf.gather_nd(input_tensor,base_idx)#batch*128*k*128
    theta=n*tf.reduce_min(tf.reduce_min(dis_xyz,axis=-1),axis=-1)
    in_xyz=tf.expand_dims(xyz,axis=2)
    if proj_type=='g':
        guassian=tf.exp(-tf.reduce_sum(tf.square(in_xyz-base_xyz),axis=-1)/tf.square(theta)) #batch*128*k
    ex_tensor=[]
    affect_feat=guassian*base_feat
    for i in range(up_ratio):
        kernel=get_weight_variable([1,1,k,1],1e-3,'weights%d'%i)
        ex_tensor.append(tf.reduce_sum(kernel*affect_feat,axis=2))
    out_tensor=tf.expand_dims(tf.concat(ex_tensor,axis=1),axis=2)#batch*512*1*128
    for i,outchannel in enumerate(mlp):
        tensor=conv2d('to_point%d'%i,tensor,outchannel,[1,1],padding='VALID')
    pt = tf.squeeze(conv2d('topt', net, 3, [1, 1],activation_func=None),[2])
    return pt,out_tensor

def gradual_rbf(tensor):
    featLen = tensor.get_shape()[-1].value
    batchNum = tf.shape(tensor)[0]
    input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])
    with tf.variable_scope('D128'):
        net128,netfeat128=decom_ful('decom_layer0',input_tensor,[512,256],[256],[256],128,128)
        net128=tf.squeeze(net128,[2])
    tf.add_to_collection('o128', net128)
    with tf.variable_scope('D512'):
        net512,netfeat512=rbf_expand(net128,netfeat128,[64,64,64])
    tf.add_to_collection('o512',net512)
    with tf.variable_scope('D2048'):
        net2048,netfeat2048=rbf_expand(net512,netfeat512,[64,64,64])
    tf.add_to_collection('o2048', net2048)
    return tf.reshape(net128, [-1, 128, 3]),tf.reshape(net512, [-1, 512, 3]),tf.reshape(net2048, [-1, 2048, 3])




def get_kneighbor(ptdata,dis_oself,k):
    kindex=tf.expand_dims(tf.nn.top_k(-dis_oself,k)[1],axis=-1)#batch*2048*k*1
    batchindex=tf.reshape(tf.constant([i for i in range(BATCH_SIZE)]),[BATCH_SIZE,1,1,1])
    index=tf.concat([tf.tile(batchindex,multiples=[1,tf.shape(kindex),k,1]),kindex],axis=-1)#batch*2048*k*2
    kneighbor=tf.gather_nd(ptdata,index)#batch*2048*k*3
    return kneighbor,index
def get_cen_kneighbor(ptdata,dis_self,k):
    kindex=tf.expand_dims(tf.nn.top_k(-dis_self,k)[1],axis=-1)#batch*2048*k*1
    batchindex=tf.reshape(tf.constant([i for i in range(BATCH_SIZE)]),[BATCH_SIZE,1,1,1])
    
    index=tf.concat([tf.tile(batchindex,multiples=[1,tf.shape(kindex)[1],k,1]),kindex],axis=-1)#batch*2048*k*2
    kneighbor=tf.gather_nd(ptdata,index)#batch*2048*k*3
    return kneighbor,index
#make k points near a point as far as possible from it
def fardis_func(dis_oself,k=6):
    kdis=-tf.nn.top_k(-dis_oself,k)[0]
    kmeandis=tf.reduce_mean(tf.reduce_sum(kdis,axis=-1)/(k-1),axis=-1)#because there is a zeros for itself
    kfarloss=-tf.reduce_mean(tf.log(kmeandis))
    return kfarloss
def equaldis_func(dis_oself,k=6):
    kdis = tf.log(-tf.nn.top_k(-dis_oself, k)[0]+1e-5) #batch*2048*k
    kmeandis=tf.expand_dims(tf.reduce_mean(kdis,axis=-1),axis=-1)
    #kdisstd=tf.reduce_max(tf.log(tf.reduce_max(kdis,axis=-1))-tf.log((tf.reduce_min(kdis,axis=-1)+1e-10)))
    kdisstd=tf.sqrt(tf.reduce_max(tf.reduce_mean(tf.square(kdis-kmeandis),axis=-1),axis=-1))
    #kdisstd=tf.sqrt(kdisvar)
    result=tf.reduce_mean(kdisstd)
    return result
#calculate normal directions of every point by guassian PCA,return batch*2048*3
def getdirection(output,dis_oself,k=16):
    kneighbor,_=get_kneighbor(output,dis_oself,k)#batch*2048*k*3
    kndis=tf.sqrt(tf.reduce_sum(tf.square(kneighbor-tf.expand_dims(output,axis=2)),axis=-1))#batch*2048*k
    radius=tf.reduce_max(kndis,axis=-1)/3 #batch*2048
    A=tf.expand_dims(output,axis=2)-kneighbor #batch*2048*k*3
    guassian=tf.exp(-tf.reduce_sum(tf.square(A),axis=-1)/tf.square(tf.expand_dims(radius,axis=-1))) #batch*2048*k
    B=tf.transpose(A,[0,1,3,2])*tf.expand_dims(guassian,axis=2) #batch*2048*3*k
    neighvar=tf.matmul(B,A) #batch*2048*3*3
    _,vecs=tf.self_adjoint_eig(neighvar)
    normalvec=vecs[:,:,:,0] #batch*2048*3
    return normalvec

def classify_loss(pred,label):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    #print(loss)
    classify_loss = tf.reduce_mean(loss)
    #tf.summary.scalar('classify loss', classify_loss)
    #tf.add_to_collection('losses', classify_loss)
    return classify_loss
def segment_loss(pred,label,smpw=1):
    """ pred: BxNxC,
        label: BxN, 
	smpw: BxN """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

#make k points neighbor have similiar directions
def smooth_func(output,dis_oself,k=8):
    normalvecs=getdirection(output,dis_oself)
    _,kindex=get_kneighbor(output,dis_oself,k)
    kneighborvecs=tf.gather_nd(normalvecs,kindex) #batch*2048*k*3
    kvecs=tf.reduce_sum(tf.expand_dims(kneighborvecs,axis=2)*tf.expand_dims(kneighborvecs,axis=3),axis=-1) #batch*2048*k*k
    kvecsmax=tf.reduce_max(tf.reduce_max(tf.sqrt(1-tf.square(kvecs)),axis=-1),axis=-1)#?
    kvecs_loss=tf.reduce_mean(tf.reduce_mean(kvecsmax,axis=-1))
    return kvecs_loss,kvecsmax
def rms_func(input,output):
    dis_i=tf.reduce_sum(tf.square(tf.expand_dims(input,axis=2)-tf.tile(tf.expand_dims(output,axis=1),multiples=[1,tf.shape(input)[1],1,1])),axis=-1)
    dis_o=tf.transpose(dis_i,[0,2,1])
    dis_ii=tf.reduce_mean(tf.reduce_min(dis_i,axis=-1))
    dis_oo=tf.reduce_mean(tf.reduce_min(dis_o,axis=-1))
    dis=tf.sqrt(tf.reduce_mean(tf.maximum(dis_ii,dis_oo)))
    return dis
def chamfer_func(input,output):
    dis_i=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(input,axis=2)-tf.tile(tf.expand_dims(output,axis=1),multiples=[1,tf.shape(input)[1],1,1])),axis=-1))
    dis_o=tf.transpose(dis_i,[0,2,1])
    dis_ii=tf.reduce_mean(tf.reduce_min(dis_i,axis=-1))
    dis_oo=tf.reduce_mean(tf.reduce_min(dis_o,axis=-1))
    dis=tf.reduce_mean(tf.maximum(dis_ii,dis_oo))

    cens=tf.reduce_mean(input,axis=1,keepdims=True)
    radius=tf.sqrt(tf.reduce_max(tf.reduce_sum((input - cens) ** 2,axis=-1),axis=-1))
    dist_norm = dis / radius

    return dis

def emd_func(pred,gt):
    batch_size = pred.get_shape()[0].value
    matchl_out, matchr_out = tf_auctionmatch.auction_match(pred, gt)
    matched_out = tf_sampling.gather_point(gt, matchl_out)
    dist = tf.sqrt(tf.reduce_sum((pred - matched_out) ** 2,axis=-1))
    dist = tf.reduce_mean(dist,axis=-1)

    cens=tf.reduce_mean(pred,axis=1,keep_dims=True)
    radius=tf.sqrt(tf.reduce_max(tf.reduce_sum((pred - cens) ** 2,axis=-1),axis=-1))
    #dist = tf.reshape((pred - matched_out) ** 2, shape=(batch_size, -1))
    #dist = tf.reduce_mean(dist, axis=1, keep_dims=True)
    #print(matched_out,dist)
    dist_norm = dist / radius

    emd_loss = tf.reduce_mean(dist_norm)
    return emd_loss
def emd_gather(xyz,ptid):
    bnum=tf.shape(xyz)[0]
    ptnum=xyz.get_shape()[1].value
    npoint=tf.shape(ptid)[-1]
    #ptids=arange(ptnum)
    #random.shuffle(ptids)
    #ptid=tf.tile(tf.constant(ptids[:npoint],shape=[1,npoint,1],dtype=tf.int32),[bnum,1,1])
    bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,npoint,1])
    idx=tf.concat([bid,tf.expand_dims(ptid,axis=-1)],axis=-1)
    new_xyz=tf.gather_nd(xyz,idx)
    return new_xyz
#def iter_emd(pred,gt):
#    num_points=tf.cast(gt.shape[1], tf.float32)
#    matchl_out, matchr_out = iter_match(pred, gt)
#    matched_out = emd_gather(gt, matchl_out)
#    
#    dist = tf.sqrt(tf.reduce_sum((pred - matched_out) ** 2,axis=-1))
#    #print(dist)
#    #cens=tf.reduce_mean(pred,axis=1,keepdims=True)
#    #radius=tf.sqrt(1e-8+tf.reduce_sum(tf.reduce_sum((pred - cens) ** 2,axis=-1),axis=1))
#    #dist = dist / radius
#    dist = tf.reduce_mean(dist)#+0.2*tf.reduce_max(dist)
#    return dist
#def val_emd(pred,gt):
#    matchl_out, matchr_out = val_match(pred, gt)
#    #matched_out = tf_sampling.gather_point(gt, matchl_out)
#    matched_out=emd_gather(gt,matchl_out)
#    dist = tf.sqrt(tf.reduce_sum((pred - matched_out) ** 2,axis=-1))
#    dist = tf.reduce_mean(dist,axis=-1)
#    return tf.reduce_mean(dist)

def chamfer_big(pcd1, pcd2):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    #dist11 = tf.reduce_mean(tf.sqrt(dist1))
    #dist22 = tf.reduce_mean(tf.sqrt(dist2))
    #w1=tf.exp(tf.sqrt(dist1))/tf.reduce_sum(tf.exp(tf.sqrt(dist1)),axis=-1,keepdims=True)
    #w2=tf.exp(tf.sqrt(dist2))/tf.reduce_sum(tf.exp(tf.sqrt(dist2)),axis=-1,keepdims=True)
    #wdist1=tf.reduce_mean(tf.reduce_sum(w1*tf.sqrt(dist1),axis=-1))
    #wdist2=tf.reduce_mean(tf.reduce_sum(w2*tf.sqrt(dist2),axis=-1))
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    #dist1 = tf.reduce_mean(dist1)
    #dist2 = tf.reduce_mean(dist2)
    return (dist1 + dist2) / 2,idx1
def fidelity_loss(pcd1,pcd2):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    #dist2 = tf.reduce_mean(tf.sqrt(dist2))
    return dist1

def earth_mover(pcd1, pcd2):
    #print(pcd1,pcd2)
    assert pcd1.shape[1] == pcd2.shape[1]
    #print(pcd1,pcd2)
    num_points = tf.cast(pcd1.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    return tf.reduce_mean(cost / num_points)
#Distance in a group as near as possible
#batch*1024*16*3
def groupin_near(ptmat):
    result=tf.reduce_sum(tf.square(ptmat),axis=-1)
    result=tf.reduce_mean(tf.reduce_mean(result,axis=-1),axis=-1)
    result=tf.reduce_mean(result)
    return result
#batch*1024*16*3
def groupin_vol(ptmat):
    result=tf.pow(tf.sqrt(tf.reduce_sum(tf.square(ptmat),axis=-1)),3)
    result=tf.reduce_mean(tf.reduce_mean(result,axis=-1),axis=-1)
    result=tf.reduce_mean(result)
    return result
#batch*1024*16*3
def groupin_cen(ptmat):
    result=tf.reduce_sum(ptmat,axis=2)#batch*1024*3
    result=tf.reduce_mean(tf.reduce_sum(tf.square(result),axis=-1),axis=[0,1])
    return result
#batch*1024*16*3
def groupin_cenon(ptmat):
    result=tf.reduce_sum(tf.square(ptmat),axis=-1)
    result=tf.reduce_mean(tf.reduce_min(result,axis=-1),axis=-1)
    result=tf.reduce_mean(result)
    return result
#Distance between different groups as far as possible
#batch*1024*16*3
def groupby_far(cenmat,ptmat,k=2):
    result=0
    bnum=tf.shape(cenmat)[0]
    ptnum=ptmat.get_shape()[1].value
    cenmat=tf.expand_dims(cenmat,axis=2)
    dismat=tf.reduce_sum(tf.square(ptmat),axis=-1)#batch*64*16
    fark,farkid=tf.nn.top_k(dismat,k)#batch*64*k,batch*64*k
    ptid=tf.tile(tf.constant([i for i in range(ptnum)],shape=[1,ptnum,1,1],dtype=tf.int32),[bnum,1,k,1])
    bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1,1]),[1,ptnum,k,1])
    idx=tf.concat([bid,ptid,tf.expand_dims(farkid,axis=-1)],axis=-1)
    farpts=tf.gather_nd(ptmat,idx)+tf.tile(cenmat,[1,1,k,1])#batch*64*k*3
    farpts=tf.concat([farpts,cenmat],axis=2)
    farpts=tf.reshape(farpts,[-1,ptnum*(k+1),3])
    kdismat=tf.reduce_sum(tf.square(tf.expand_dims(farpts,axis=2)-tf.expand_dims(farpts,axis=1)),axis=-1)+100*tf.tile(tf.expand_dims(tf.eye(num_rows=(k+1)*ptnum),axis=0),[bnum,1,1])#batch*128*128
    kdis=tf.exp(-tf.reduce_mean(tf.reduce_min(kdismat,axis=-1),axis=[0,1]))
    return kdis
#sampling from generated points similiar with first layer
def sample_loss(sourcepts,despts,method='f'):
    ptnum=sourcepts.get_shape()[1].value
    _,sampts=sampling(ptnum,despts,use_type=method)
    loss=earth_mover(sourcepts,sampts)
    return loss
def grouping_pts(radius,ksample,rawpts,cenpts,knn=False):
    if knn:
        _,idx = tf_grouping.knn_point(ksample, rawpts, cenpts)
    else:
        idx,_ = tf_grouping.query_ball_point(radius, ksample, rawpts, cenpts)
    grouped_xyz = tf_grouping.group_point(rawpts, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(cenpts, 2), [1,1,ksample,1]) # translation normalization
    return grouped_xyz
def local_loss(radius,ksample,rawpts,genpts,cenpts,use_type='c',knn=True):
    #print(rawpts,cenpts)
    raw_group=grouping_pts(radius,ksample,rawpts,cenpts,knn)
    gen_group=grouping_pts(radius,ksample,genpts,cenpts,knn)#batch*npoint*k*3
    if use_type=='e':
        loss=earth_mover(tf.reshape(raw_group,[-1,ksample,3]),tf.reshape(gen_group,[-1,ksample,3]))
    elif use_type=='c':
        loss=chamfer_big(tf.reshape(raw_group,[-1,ksample,3]),tf.reshape(gen_group,[-1,ksample,3]))
        #in_mat=tf.reduce_sum(tf.square(tf.expand_dims(raw_group,axis=3)-tf.expand_dims(gen_group,axis=2)),axis=-1)# batch*n*k*k
        #out_mat=tf.transpose(in_mat,[0,1,3,2])
        #in_loss=tf.reduce_mean(tf.reduce_mean(tf.reduce_min(in_mat,axis=-1),axis=-1),axis=-1) #batch*n*k*k find sum loss of all locals
        #out_loss=tf.reduce_mean(tf.reduce_mean(tf.reduce_min(out_mat,axis=-1),axis=-1),axis=-1)
        #loss=tf.sqrt(tf.reduce_mean(tf.maximum(in_loss,out_loss)))
    loss=0.5*loss+0.5*chamfer_big(rawpts,genpts)
    return loss
#idx:batch*ptnum
#pointcloud:batch*16384*3
#output:batch*ptnum*3
def get_pts(pointcloud,idx):
    ptnum=pointcloud.get_shape()[1].value
    batchidx=tf.reshape(tf.range(tf.shape(pointcloud)[0]),[-1,ptnum,1])
    idmat=tf.concat([batchidx,tf.expand_dims(idx,axis=-1)],axis=-1)
    newptcloud=tf.gather_nd(pointcloud,idmat)
    return newptcloud
#ptcens:batch*64*3
#rawpts:batch*1024*3
#outvec:batch*ptnum*16*3
def zero_groupin_near(ptcens,idx,rawpts,outvec): 
    lastnum=ptcens.get_shape()[1].value
    up_ratio=outvec.get_shape()[-2].value
    rawpts1=get_pts(rawpts,idx)
    rawvec=tf.reshape(rawpts,[-1,lastnum,up_ratio,3])-tf.expand_dims(ptcens,axis=2)#batch*ptnum*16*3
    vec1=tf.reduce_mean(tf.reduce_sum(tf.square(rawvec),axis=-1),axis=-1)#batch*ptnum
    vec2=tf.reduce_mean(tf.reduce_sum(tf.square(outvec),axis=-1),axis=-1)
    result=tf.reduce_mean(tf.square(vec1-vec2),axis=[0,1])
    return result
#batch*64*3
def refine_near(rawpts,repts):
    refdis=tf.reduce_mean(tf.reduce_sum(tf.square(repts-rawpts),axis=-1),axis=[0,1]) #batch*64*3
    return refdis
   
def zero_groupnear(ptcens,rawpts,outmat,name='balance_parameter'):
    _, _,dist,_ = tf_nndistance.nn_distance(ptcens, rawpts)
    inval=tf.reduce_mean(dist)
    outval=groupin_near(outmat)
    #inval=tf.get_variable(name=name,shape=[1],initializer=tf.contrib.layers.xavier_initializer())
    #result=tf.abs(1-outval/(inval+1e-5))*outval
    result=tf.nn.relu(outval-0.4*inval)
    #result=tf.nn.relu(outval-tf.squeeze(inval))+tf.square(inval)
    return result
def multi_re_loss(pointcloud,outlist,weights=[1,1,1],funclist=[earth_mover,chamfer_big,chamfer_big],alpha0=1,alpha=1,alpha2=1):
    #loss_list=[]
    #for i in range(len(outlist)):
    #    func=funclist[i]
    #    ptnum=outlist[i].get_shape()[1].value
    #    _,points=sampling(ptnum,pointcloud)
    #    loss_list.append(func(points,outlist[i]))
    loss=0
    ptnum=outlist[0].get_shape()[1].value
    _,points1=sampling(ptnum*16,pointcloud,use_type='f')
    _,points=sampling(ptnum,points1,use_type='f')

    #loss+=alpha0*earth_mover(tf.get_collection('pts_tar1')[0],outlist[0])
    loss+=alpha0*earth_mover(points,outlist[0])
    loss2,ptid2=chamfer_big(points1,outlist[1])
    loss+=alpha2*loss2
    #loss+=alpha2*(chamfer_big(tf.get_collection('pts_tar2')[0],outlist[1])\
    #             +chamfer_big(tf.get_collection('pts_tar2')[0],tf.get_collection('tar_tar2')[0]))
    #loss+=alpha2*local_loss(0.5,16,points1,outlist[1],points)
    #loss+=0.01*sample_loss(points,outlist[1],'f')
    loss3,ptid3=chamfer_big(pointcloud,outlist[-1])
    loss+=alpha*loss3
    #loss+=alpha*(chamfer_big(pointcloud,outlist[-1])+\
    #             chamfer_big(pointcloud,tf.get_collection('tar_tar3')[0]))
    #loss+=alpha2*local_loss(0.5,64,pointcloud,outlist[-1],points1)
    #loss+=0.01*sample_loss(points,outlist[-1],'f')
    #loss+=0.001*groupby_far(tf.get_collection('o128')[0],tf.get_collection('decode_cell64')[0],k=1)
    #loss+=0.001*groupby_far(tf.get_collection('o512')[0],tf.get_collection('decode_cell1024')[0],k=1)
    #loss_d1=groupin_near(tf.get_collection('decode_cell64')[0])
    #loss+=alpha0*loss1
    loss_d1=0.1*zero_groupnear(points,points1,tf.get_collection('decode_cell64')[0])
    #loss+=alpha2*loss2
    #loss_d2=groupin_near(tf.get_collection('decode_layer1024')[0])
    loss_d2=0.1*zero_groupnear(points1,pointcloud,tf.get_collection('decode_cell1024')[0])
    #loss+=alpha*loss3
    loss=loss+loss_d1+loss_d2
    #tf.add_to_collection('loss_base',tf.stack([loss1,loss2,loss3],axis=0))
    tf.add_to_collection('loss64',loss_d1)
    #tf.add_to_collection('loss1024',loss_d2)
    tf.add_to_collection('loss1024',groupin_near(tf.get_collection('decode_cell64')[0]))
    #loss+=0.1*groupin_cen(tf.get_collection('decode_cell64')[0])
    #loss+=zero_groupin_near(tf.get_collection('cen1')[0],ptid2,points1,tf.get_collection('decode_cell64')[0])
    #loss+=0.1*groupin_near(tf.get_collection('decode_cell1024')[0])
    #loss+=0.1*groupin_near(tf.get_collection('refine_layer1024')[0])
    #tf.add_to_collection('loss1024',groupin_near(tf.get_collection('decode_cell1024')[0]))
    #loss+=0.1*groupin_cen(tf.get_collection('decode_cell1024')[0])

    #loss+=0.05*groupin_near(tf.get_collection('decode_cell1024')[0]+tf.reshape(tf.get_collection('decode_cell64')[0],[-1,1024,1,3]))
    #loss+=0.01*refine_near(outlist[0],tf.get_collection('cen1')[0])
    #loss+=0.01*refine_near(outlist[1],tf.get_collection('cen2')[0])
    #loss+=zero_groupin_near(tf.get_collection('cen2')[0],ptid3,pointcloud,tf.get_collection('decode_cell1024')[0])
    #loss+=0.05*groupin_near(tf.get_collection('decode_cell_init_grid64')[0])
    #loss+=0.05*groupin_near(tf.get_collection('decode_cell_init_grid1024')[0])
    #loss+=0.05*groupin_near(tf.get_collection('decode_cell_fold_grid64')[0])
    #loss+=0.05*groupin_near(tf.get_collection('decode_cell_fold_grid1024')[0])
    #for i in range(len(loss_list)):
    #    loss+=weights[i]*loss_list[i]
    #return loss_list
    return loss
def chamfer_local(ptgt,ptout,cenum=128,knum=16,ctype='r'):
    _,cens=sampling(cenum,ptgt,ctype)
    idx=get_topk(cens,ptgt,knum=knum)
    ptkgt=tf.gather_nd(ptgt,idx)-tf.expand_dims(cens,axis=2)#batch*n*k*c

    idout=get_topk(cens,ptout,knum=knum)
    ptkout=tf.gather_nd(ptout,idout)-tf.expand_dims(cens,axis=2)#batch*n*k*c
    ch=ptgt.get_shape()[-1].value
    dist=earth_mover(tf.reshape(ptkgt,[-1,knum,ch]),tf.reshape(ptkout,[-1,knum,ch]))

    #dismat=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(ptkgt,axis=2)-tf.expand_dims(ptkout,axis=3)),axis=-1))#batch*n*k*k
    #dist1=tf.reduce_mean(tf.reduce_min(dismat,axis=-1),axis=-1)#batch*n
    #dist2=tf.reduce_mean(tf.reduce_min(dismat,axis=-2),axis=-1)#batch*n
    #dist=tf.reduce_mean((dist1+dist2)/2)
    return dist
def multi_chamfer_func(cen,input,output,n,k,r=0.2,theta1=0.5,theta2=0.5,use_frame=True,use_r=False,use_all=False):
    if use_all:
        in_cen=input
    else:
        in_cen=tf.get_collection('i'+str(n))[0]
    #in_cen=tf.gather_nd(input,in_index)
    if use_r:
        out_kneighbor=tf.gather_nd(output,point_choose.local_cen_devide(n,k,r,output,in_cen,batch=batch_size))-tf.expand_dims(in_cen,axis=2)
        in_kneighbor=tf.gather_nd(input,point_choose.local_cen_devide(n,k,r,input,in_cen,batch=batch_size))-tf.expand_dims(in_cen,axis=2)
    else:
        dis_ci=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(in_cen,axis=2)-tf.expand_dims(input,axis=1)),axis=-1))#batch*n*2048
        dis_co=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(in_cen,axis=2)-tf.expand_dims(output,axis=1)),axis=-1))#batch*n*2048
        out_kneighbor=get_cen_kneighbor(output,dis_co,k)[0]-tf.expand_dims(in_cen,axis=2) #batch*n*k*3
        in_kneighbor=get_cen_kneighbor(input,dis_ci,k)[0]-tf.expand_dims(in_cen,axis=2) #batch*n*k*3

    in_mat=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(in_kneighbor,axis=3)-tf.expand_dims(out_kneighbor,axis=2)),axis=-1))# batch*n*k*k
    out_mat=tf.transpose(in_mat,[0,1,3,2])
    in_loss=tf.reduce_mean(tf.reduce_mean(tf.reduce_min(in_mat,axis=-1),axis=-1),axis=-1) #batch*n*k*k find sum loss of all locals
    out_loss=tf.reduce_mean(tf.reduce_mean(tf.reduce_min(out_mat,axis=-1),axis=-1),axis=-1)
    local_loss=tf.reduce_mean(tf.maximum(in_loss,out_loss))
    if use_frame:
        #frame_loss=regression_func(in_cen,cen)
        frame_loss=chamfer_func(input,output)
        return theta1*frame_loss+theta2*local_loss
    else:
        return theta2*local_loss
#make codeword more sparse
def codelimit_func(codeword):
    #return tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(codeword),axis=-1)))
    codeword=tf.cast(codeword,tf.float32)
    return tf.reduce_mean(tf.reduce_sum(codeword,axis=-1))

def get_gradients(loss,scope):
    with tf.variable_scope(scope,reuse=True):
        wlayer = tf.gradients(loss, tf.get_variable(name='weights'))
    tf.summary.histogram(scope+'/gradients',wlayer)
    return wlayer
def get_gen(path,batch_size=BATCH_SIZE,input_ptnum=INNUM,output_ptnum=PTNUM,is_training=True):
    df, num = lmdb_dataflow(path,batch_size,input_ptnum,output_ptnum,is_training)
    gen = df.get_data()
    return gen,num
def train_one_batch(sess,ops,train_writer,batch):
    ids,batch_point,npts,output_point=next(ops['train_gen'])
    #print(shape(batch_point))
    #getdata.saveh5data(batch_point,'data','input.h5')
    #getdata.saveh5data(output_point,'data','output.h5')
    feed_dict = {ops['pointcloud_pl']: batch_point,ops['gt_pl']:output_point,ops['type_pl']:'p'}
    resi = sess.run([ops['trainstep'],ops['loss'],ops['iteremd'],ops['cd'],ops['zhengze']], feed_dict=feed_dict)
    #resi = sess.run([ops['loss']], feed_dict=feed_dict)
    #print('trainstep complete')
    code1,code2,code3=sess.run([ops['code1'],ops['code2'],ops['code3']],feed_dict=feed_dict)
    #loss64,loss1024=sess.run([ops['loss64'],ops['loss1024']],feed_dict=feed_dict)
    #loss1,loss2,loss3=sess.run([ops['loss_base'][0],ops['loss_base'][1],ops['loss_base'][2]],feed_dict=feed_dict)
    code1=code1[0]
    code2=code2[0]
    code3=code3[0]
    if (batch+1) % 500 == 0:
        #code1,code2,code3=sess.run([ops['code1'],ops['code2'],ops['code3']],feed_dict=feed_dict)
        #loss64,loss1024=sess.run([ops['loss64'],ops['loss1024']],feed_dict=feed_dict)
        #loss1,loss2,loss3=sess.run([ops['loss_base'][0],ops['loss_base'][1],ops['loss_base'][2]],feed_dict=feed_dict)
        #code1=code1[0]
        #code2=code2[0]
        #code3=code3[0]
        result = sess.run(ops['merged'], feed_dict=feed_dict)
        train_writer.add_summary(result, batch)
        print('epoch: %d'%(batch*BATCH_SIZE//ops['train_num']+1),'batch: %d' %batch)
        print('loss: ',resi[1])
        print('decode_cell regularization: ', resi[-1])
        print('max of code1 first: %f'%max(code1),'code1 nonzero num:%d'%len(code1[code1!=0]))
        print('max of code2 first: %f'%max(code2),'code2 nonzero num:%d'%len(code2[code2!=0]))
        print('max of code3 first: %f'%max(code3),'code3 nonzero num:%d'%len(code3[code3!=0]))
        #print('loss64: ',loss64)
        #print('loss64: ',loss1024)
        print('loss: ',resi[1])
        print('emd loss',resi[2])
        print('cd loss',resi[3])
        #print('loss2: ',loss2)
        #print('loss3: ',loss3)

def eval_one_batch(sess,ops):
    chamferlist=[]
    emdlist=[]
    itertime=ops['valid_num']//EVAL_SIZE
    print('evaluate begin_________')
    for i in range(itertime):
        ids,batch_point,npts,output_point=next(ops['valid_gen'])
        feed_dict = {ops['pointcloud_pl']: batch_point,ops['gt_pl']:output_point,ops['type_pl']:'p'}
        resi_chamfer,resi_emd = sess.run([ops['chamfer_loss'],ops['emd_loss']],feed_dict=feed_dict)
        chamferlist.append(resi_chamfer)
        emdlist.append(resi_emd)
        #print('batch: %d' %i)
        #print('chamfer loss: %f'%resi_chamfer)
        #print('emd loss: %f'%resi_emd)
    print('mean chamfer loss: %f'%mean(chamferlist))
    print('mean emd loss: %f'%mean(emdlist))
    print('evalueate end__________')
    return mean(emdlist),mean(chamferlist)

def train():
   
    pointcloud_pl=tf.placeholder(tf.float32,[None,3000,3],name='pointcloud_pl')
    gt_pl=tf.placeholder(tf.float32,[None,PTNUM,3],name='gt_pl')
    type_pl=tf.placeholder(tf.string,shape=(),name='type_pl')
    #model_num=8
    #word=re_encoder(pointcloud_pl)
    #out1,out2,out3=re_decoder(word)
    #print('************')
    out1,out2,out3,out4=full_process(pointcloud_pl)
    
    train_gen,train_num=get_gen('../dense_data/train.lmdb')
    #print('train_num',train_num)
    valid_gen,valid_num=get_gen('../dense_data/valid.lmdb',batch_size=EVAL_SIZE,is_training=False)
    #sub3=model_substract(gt_pl,pointcloud_pl,min_dist=10,max_iter=20,bsize=BATCH_SIZE)
    sub3=gt_pl
    print('*****')
    layernum1=out1.get_shape()[1].value
    layernum2=out2.get_shape()[1].value
    _,sub1=sampling(layernum1,sub3,use_type='f')
    _,sub2=sampling(layernum2,sub3,use_type='f')
    
    global_step=tf.Variable(0,trainable=False)

    #learning_rate = tf.train.exponential_decay(0.0005, global_step,12500, 0.8,staircase=True, name='lr')
    #alpha0 = tf.maximum(learning_rate, 1e-6)

    alpha0 = tf.train.piecewise_constant(global_step, [50000,100000,150000,200000],
                                        [0.0005, 0.0002,0.0002,0.0001,0.00001], 'alpha_op')
    alpha1 = tf.train.piecewise_constant(global_step, [50000,150000],
                                        [0.01, 0.01,0.001], 'alpha_op')
    #alpha = tf.train.piecewise_constant(global_step, [10000, 20000, 50000, 100000],
    #                                    [0.001, 0.01, 0.1, 0.1, 0.5], 'alpha_op')
    #alpha2 = tf.train.piecewise_constant(global_step, [10000, 20000, 50000, 100000],
    #                                    [0.01, 0.1, 0.5, 0.5, 1.0], 'alpha_op')
    #alpha0 = tf.train.piecewise_constant(global_step, [10000, 20000, 50000, 100000],
    #                                    [1.0, 1.0, 1.0, 1.0, 1.0], 'alpha_op')
    #alpha = tf.train.piecewise_constant(global_step, [10000, 20000, 50000, 100000],
    #                                    [0.001, 0.01, 0.1, 0.1, 0.1], 'alpha_op')
    #alpha2 = tf.train.piecewise_constant(global_step, [10000, 20000, 50000, 100000],
    #                                    [0.1, 0.1, 0.5, 0.5, 0.5], 'alpha_op')
    
    #alpha0=1.0
    #alpha=1.0
    #alpha2=1.0
    #train_gen,train_num=get_gen('train/train.lmdb')
    #print('train_num',train_num)
    #valid_gen,valid_num=get_gen('../dense_data/valid.lmdb',batch_size=EVAL_SIZE,is_training=False)
    zhengze=tf.add_n(tf.get_collection('losses'))
    #zhengze=sum([tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(v) for v in tf.trainable_variables() if 'decode_cell' in v.name])
    #loss=multi_re_loss(gt_pl,out,alpha0=alpha0,alpha=alpha,alpha2=alpha2)#+0.01*zhengze

    #memory=init_memory('mem',[model_num,PTNUM,3])
    #loss_mem=memory_loss(memory,gt_pl)
    chamfer_loss,ptid=chamfer_big(gt_pl,out4)
    emd_loss=earth_mover(gt_pl,out4)
    #iteremd=earth_mover(gt_pl,out)
    #iteremd=tf.minimum(iter_emd(gt_pl,out3),iter_emd(out3,gt_pl))
    #iteremd2=tf.minimum(iter_emd(gt_pl,out4),iter_emd(out4,gt_pl))
    #refinemove1=tf.get_collection('refine_layer64')[0]
    #refinemove2=tf.get_collection('refine_layer1024')[0]
    refinemove3=tf.get_collection('refine_layer_final16384')[0]
    fd3=fidelity_loss(pointcloud_pl,out3)
    moveloss=tf.reduce_mean(tf.reduce_sum(tf.square(refinemove3),axis=-1))

    #iteremd=iter_emd(gt_pl,out3)
    #iteremd2=iter_emd(gt_pl,out4)

    #valemd=val_emd(gt_pl,out3)
    #valemd2=val_emd(gt_pl,out4)

    cd1=earth_mover(sub1,tf.get_collection('points1')[0])
    ccd1,_=chamfer_big(sub1,tf.get_collection('points1')[0])
    cd2=earth_mover(sub2,tf.get_collection('points2')[0])
    ccd2,_=chamfer_big(sub2,tf.get_collection('points2')[0])
    cd3,_=chamfer_big(gt_pl,out3)
    cd4,_=chamfer_big(gt_pl,out4)
    recd3=re_chamfer(gt_pl,out3,part=8)
    recd4=re_chamfer(gt_pl,out4,part=8)
    ccd3=chamfer_local(gt_pl,out3,cenum=256,knum=16,ctype='f')

    #out3=sampling(16384,out3,'r')[1]
    #remd=earth_mover(sampling(2048,out3,use_type='f')[1],sampling(2048,gt_pl,use_type='f')[1])

    subgt=naive_substract(gt_pl,pointcloud_pl,bsize=BATCH_SIZE)
    subout=naive_substract(out4,pointcloud_pl,bsize=BATCH_SIZE)
    subcd4,_=chamfer_big(subgt,subout)
    #fd3=fidelity_loss(gt_pl,out3)
    #emdvar=earth_mover(sub1,tf.get_collection('startvar')[0])
    #cdvar,_=chamfer_big(gt_pl,tf.get_collection('varpts')[0])
    #secloss=0.5*(tf.add_n(tf.get_collection('sec2'))+tf.add_n(tf.get_collection('sec3')))
    #repu=get_repulsion_loss4(sampling(2048,out3,use_type='r')[1],out3)
    #loss=(cd1+0.5*cd2)+0.2*cd3+0.03*repu+0.2*cd4+0.1*moveloss#+secloss
    #cd1=ccd1+0.2*cd1
    #cd2=ccd2+0.2*cd2
    loss=0.2*(cd1+cd2)+cd3+fd3+cd4+0.2*recd3+0.1*moveloss
    #loss=iteremd+iteremd2
    #layernum1=64
    #layernum2=1024
    loss_d1=0.05*zero_groupnear(sub1,sub2,tf.get_collection('decode_cell'+str(layernum1))[0],name='balance_para1')
    loss_d2=0.05*zero_groupnear(sub2,gt_pl,tf.get_collection('decode_cell'+str(layernum2))[0],name='balance_para2')
    loss_dec=tf.add_n(tf.get_collection('decfactor'))
    loss=loss+loss_d1+loss_d2+alpha1*loss_dec
    #loss=loss+alpha1*loss_dec
    #cdlist=[cd1,cd2,cd3]
    #loss=[]
    #losslist[-1]+=0.01*zhengze
    #for i in range(len(losslist)):
    #    loss.append(tf.expand_dims(losslist[i],axis=-1))
    #loss=tf.concat(loss,axis=-1)

    #trainvars=tf.GraphKeys.TRAINABLE_VARIABLES
    var=[v for v in tf.global_variables()]
    myvar=[v for v in var if v.name.split(':')[0].split('_')[0]!='subvar']
    #var1=tf.get_collection(trainvars,scope='mem')
    #print(var1)
    trainstep=tf.train.AdamOptimizer(learning_rate=alpha0).minimize(loss,var_list=myvar, global_step=global_step)
    #pretrainstep=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_mem, global_step=global_step,var_list=var1)
    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(max_to_keep=20)
        sess.run(tf.global_variables_initializer())
        print('im here')
        if os.path.exists('./modelvv_recon/checkpoint'):
            print('here load')
            saver.restore(sess, tf.train.latest_checkpoint('./modelvv_recon/'))

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/", sess.graph)
        print('here,here')
        ops={'pointcloud_pl':pointcloud_pl,
             'gt_pl':gt_pl,
             'type_pl':type_pl,
             'loss':[cd1,cd2,loss_d1,loss_d2],
             'trainstep':trainstep,
             'out':out4,
     #        'loss_mem':loss_mem,
             'zhengze':zhengze,
             'merged':merged,
             'train_gen':train_gen,
             'valid_gen':valid_gen,
             'chamfer_loss':chamfer_loss,
             'emd_loss':emd_loss,
             'iteremd':tf.get_collection('decfactor'),
             'cd':cd3+cd4,
             'train_num':train_num,
             'valid_num':valid_num,
             'code1':tf.squeeze(tf.get_collection('code1')[0]),
             'code2':tf.squeeze(tf.get_collection('code2')[0]),
             'code3':tf.squeeze(tf.get_collection('code3')[0]),
      #       'loss64':tf.get_collection('loss64')[0],
      #       'loss1024':tf.get_collection('loss1024')[0]
             #'loss_base':tf.get_collection('loss_base')[0]
            }
       # train_one_batch(sess,ops,writer,0)
        bestemd=100.0
        bestcd=100.0
        #bestemd=0.0669
        #bestcd=0.0108
        for i in range(BATCH_ITER_TIME):
            train_one_batch(sess,ops,writer,i)
            if (i+1)%20000==0:
                save_path = saver.save(sess, './modelvv_recon/model',global_step=i)
                meanemd,meancd=eval_one_batch(sess,ops)
                #if meanemd<bestemd and meancd<bestcd:
                if meancd<bestcd:
                    bestemd=meanemd
                    bestcd=meancd
                    os.system('rm -r ./bestrecord')
                    os.system('mkdir ./bestrecord')
                    os.system('cp -r ./modelvv_recon ./bestrecord')
                    print('record bestsofar: ',bestemd,bestcd)
if __name__=='__main__':
    train()
