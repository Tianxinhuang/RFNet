# -*- coding: utf-8 -*-

import tensorflow as tf
from numpy import *
import os
import random

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

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
BATCH_ITER_TIME=300000
REGULARIZATION_RATE=0.00001
BATCH_SIZE=32
EVAL_SIZE=4
INNUM=3000
PTNUM=16384
FILE_NUM=6
os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf.set_random_seed(1)
def get_weight_variable(shape,stddev,name,regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)):
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

def sampling(npoint,xyz,use_type='f'):
    if use_type=='f':
        idx=tf_sampling.farthest_point_sample(npoint, xyz)
        new_xyz=tf_sampling.gather_point(xyz,idx)
    elif use_type=='r':
        bnum=tf.shape(xyz)[0]
        ptnum=xyz.get_shape()[1].value
        ptids=arange(ptnum)
        ptids=tf.random_shuffle(ptids,seed=None)
        ptidsc=ptids[:npoint]
        ptid=tf.cast(tf.tile(tf.reshape(ptidsc,[-1,npoint,1]),[bnum,1,1]),tf.int32)
        #ptid=tf.tile(tf.constant(ptidsc,shape=[1,npoint,1],dtype=tf.int32),[bnum,1,1])

        bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,npoint,1])
        idx=tf.concat([bid,ptid],axis=-1)
        new_xyz=tf.gather_nd(xyz,idx)
    return idx,new_xyz
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
        input_info=input_tensor
        if state_tensor is not None:
            #new_state=conv2d('state_trans',state_tensor,128,[1,1],padding='VALID',use_bnorm=use_bnorm)#batch*2048*1*128
            state_info=tf.tile(state_tensor,multiples=[1,tf.shape(input_tensor)[1],1,1])#batch*2048*1*128
            new_state=tf.concat([input_info,state_info],axis=-1)#batch*2048*1*256
            #new_state=input_info*state_info
        else:
            new_state=input_info
        for i,outchannel in enumerate(mlp):
            new_state=conv2d('state%d'%i,new_state,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        new_state=tf.reduce_max(conv2d('state_end',new_state,state_len,[1,1],padding='VALID',use_bnorm=use_bnorm),axis=1)
        new_state=tf.expand_dims(new_state,axis=1)
        codeout=new_state
        for i,outchannel in enumerate(mlpout):
            codeout=conv2d('codemlp%d'%i,codeout,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
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

def recover_cell(scope,input_tensor,con_tensor,mlp1,mlp2,mlpout=[128,128],reuse=False,use_bnorm=False):
    with tf.variable_scope(scope,reuse=reuse):
        tensor=input_tensor
        tensor=tf.concat([tf.tile(tensor,multiples=[1,tf.shape(con_tensor)[1],1,1]),con_tensor],axis=-1)
        for i,outchannel in enumerate(mlp2):
            tensor=conv2d('recover2%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        tensor=tf.reduce_max(tensor,axis=1,keepdims=True)
        tensor=conv2d('recover2out%d'%i,tensor,mlp2[-1],[1,1],padding='VALID',use_bnorm=use_bnorm,activation_func=None)
    return tensor
def merge_layer(rawpts,newpts,decfactor,knum=16):
    npoint=newpts.get_shape()[1].value
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(rawpts, newpts)
    grouped_xyz = tf_grouping.group_point(rawpts, tf.expand_dims(idx2,axis=-1)) # (batch_size, npoint_newpts, knum, 3)
    dismat=tf.reduce_sum(tf.square(grouped_xyz-tf.expand_dims(newpts,axis=2)),axis=-1,keepdims=True)
    ratio=tf.exp(-dismat/(1e-8+tf.square(decfactor)))
    refine_pts=newpts+tf.reduce_sum(ratio*(grouped_xyz-tf.expand_dims(newpts,axis=2)),axis=2)
    return refine_pts
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
def full_process(pointcloud):
    point_input=tf.expand_dims(pointcloud,axis=2)
    ptnum=16384
    statelen=256
    state0=global_mlp('init_mlp',point_input,[64,128,statelen])
    
    codeout1,state=encode_cell('cell',point_input,state0,mlp=[256,384],mlpout=[256,256],state_len=statelen,code_len=256,use_bnorm=False)

    codeout1=recover_cell('recover1',codeout1,point_input,mlp1=[256,256],mlp2=[256,256],use_bnorm=False)

    points1,dstate=init_move_layer(sampling(32,pointcloud,use_type='f')[1],codeout1,state_len=128,use_bnorm=False)
    partfeat=global_mlp('part_mlp',tf.expand_dims(tf.concat([pointcloud,points1],axis=1),axis=2),[64,128,statelen])
    points0,dstate0=init_decode_layer('init_cell',feat_trans(tf.concat([partfeat,codeout1],axis=-1)),None,ptnum=32,mlp=[256,256],mlp1=[128],mlp2=[256,256],state_len=128,use_bnorm=False)#batch*128*3,batch*128*1*218
    points1,dstate=tf.concat([points0,points1],axis=1),tf.concat([dstate0,dstate],axis=1)
    

    tf.add_to_collection('points1', points1)
    decfactor0=tf.get_variable(name='decline_factor0',shape=[1],initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection('decfactor',tf.square(decfactor0))
    points1=merge_layer(pointcloud,points1,decfactor0,knum=1)
    points1,dstate=refine_layer('refine_layer1',points1,feat=codeout1,feat2=dstate,use_bnorm=False)

    point_input=tf.concat([tf.expand_dims(pointcloud,axis=2),tf.expand_dims(points1,axis=2)],axis=1)
    codeout2,state=encode_cell('cell',point_input,state,mlp=[256,384],mlpout=[256,256],state_len=statelen,code_len=256,reuse=True,use_bnorm=False)
    codeout2=recover_cell('recover2',codeout2,point_input,mlp1=[256,256],mlp2=[256,256],use_bnorm=False)
    codeout2=codeout1+codeout2
    points2,dstate=decode_cell('decode_cell',codeout2,points1,dstate,None,mlp1=[128,64],up_ratio=16,state_len=128,reuse=False,use_bnorm=False)
    tf.add_to_collection('points2', points2)
    
    decfactor1=tf.get_variable(name='decline_factor1',shape=[1],initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection('decfactor',tf.square(decfactor1))
    points2=merge_layer(pointcloud,points2,decfactor1,knum=1)
    points2,dstate=refine_layer('refine_layer2',points2,feat=codeout2,feat2=dstate,reuse=False,use_bnorm=False)

    point_input=tf.concat([tf.expand_dims(pointcloud,axis=2),tf.expand_dims(points2,axis=2)],axis=1)
    codeout3,state=encode_cell('cell',point_input,state,mlp=[256,384],mlpout=[256,256],state_len=statelen,code_len=256,reuse=True,use_bnorm=False)
    codeout3=recover_cell('recover3',codeout3,point_input,mlp1=[256,256],mlp2=[256,256],use_bnorm=False)
    codeout3=codeout2+codeout3
    points3,dstate=decode_cell('decode_cell',codeout3,points2,dstate,None,mlp1=[128,64],up_ratio=16,state_len=128,reuse=True,use_bnorm=False)
    points_final=points3
    decfactor=tf.get_variable(name='decline_factor',shape=[1],initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection('decfactor',tf.square(decfactor))


    points_final=merge_layer(pointcloud,points_final,decfactor,knum=1)
    points_final,state=refine_layer('refine_layer_final',points_final,feat=codeout3,feat2=dstate,reuse=False,use_bnorm=False)
    tf.add_to_collection('o2048', points3)
    tf.add_to_collection('code1', codeout1)
    tf.add_to_collection('code2', codeout2)
    tf.add_to_collection('code3', codeout3)
    return points1,points2,points3,points_final
#batch*1*1*128,batch*1*1*128
def init_decode_layer(scope,input_tensor,state_tensor,ptnum=128,mlp=[256,256],mlp1=[64],mlp11=[64],mlp2=[128,128],state_len=128,use_bnorm=False):
    with tf.variable_scope(scope):
        if state_tensor is not None:
            input_info=conv2d('input_trans',input_tensor,128,[1,1],padding='VALID',use_bnorm=use_bnorm)
            state_info=conv2d('state_trans',state_tensor,128,[1,1],padding='VALID',use_bnorm=use_bnorm)
            new_state=tf.concat([input_info,state_info],axis=-1)
        else:
            new_state=conv2d('input_trans',input_tensor,256,[1,1],padding='VALID',use_bnorm=use_bnorm)
        for i,outchannel in enumerate(mlp):
            new_state=conv2d('basic_state%d'%i,new_state,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        points_out=new_state
        points_out=conv2d('points_out',points_out,3*ptnum+12,[1,1],padding='VALID',activation_func=None)
        transmat,movemat=points_out[:,:,:,-12:-3],points_out[:,:,:,-3:]
        transmat,movemat=tf.reshape(transmat,[-1,3,3]),tf.reshape(movemat,[-1,1,3])
        points_out=points_out[:,:,:,:-12]
        points_out=tf.reshape(tf.nn.tanh(points_out),[-1,ptnum,3])
        points_out=tf.matmul(points_out,transmat)+movemat
       
        state_out=conv2d('state_out',new_state,ptnum*16,[1,1],padding='VALID',use_bnorm=use_bnorm)
        state_out=tf.reshape(state_out,[-1,ptnum,1,16])
        state_out=tf.concat([state_out,tf.tile(new_state,[1,ptnum,1,1])],axis=-1)
        for i,outchannel in enumerate(mlp2):
            state_out=conv2d('state%d'%i,state_out,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        state_out=conv2d('state_outo',state_out,state_len,[1,1],padding='VALID',use_bnorm=use_bnorm)
        #batch*128*1*128,batch*128*1*3
    return points_out,state_out
def refine_layer(scope,ptcoor,feat,feat2,mlp=[128,64,64],mlp2=[128,128],mlpself=[128,128],reuse=False,no_feat=False,use_bnorm=False):
    with tf.variable_scope(scope,reuse=reuse):
        coor=tf.expand_dims(ptcoor,axis=2)
        featself=None
        ptnum=ptcoor.get_shape()[1].value
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
        input_info=conv2d('input_trans',input_info,256,[1,1],padding='VALID',use_bnorm=use_bnorm)
        state_info=conv2d('state_trans',state_tensor,128,[1,1],padding='VALID',use_bnorm=use_bnorm)
        if lastcode is not None:
            new_state=tf.concat([input_info,state_info,tf.tile(lastcode,[1,ptnum,1,1])],axis=-1)
        else:
            new_state=tf.concat([input_info,state_info],axis=-1)
        #new_state=tf.concat([input_info,state_info,state_shuffle],axis=-1)
        #new_state=tf.tile(input_info,[1,ptnum,1,1])*state_info
        newstatelist=[]
        for i,outchannel in enumerate(mlp):
            new_state=conv2d('basic_state%d'%i,new_state,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        points_out=new_state#batch*64*1*256
        print('***************',points_out,new_state)
        for i,outchannel in enumerate(mlp1):
            points_out=conv2d('points%d'%i,points_out,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        #points_out=conv2d('points_out',points_out,3*up_ratio,[1,1],padding='VALID',activation_func=None)
        points_out=conv2d('points_out',points_out,3*up_ratio,[1,1],padding='VALID',activation_func=tf.nn.tanh)

        points_move=tf.reshape(points_out,[-1,ptnum,up_ratio,3])
        tf.add_to_collection(scope+str(ptnum),points_move)
        #print(center,points_out) 
        points_out=tf.tile(tf.expand_dims(center,axis=2),[1,1,up_ratio,1])+points_move
        points_out=tf.reshape(points_out,[-1,ptnum*up_ratio,3])
        new_state=tf.concat([new_state,tf.tile(input_tensor,[1,ptnum,1,1])],axis=-1)
        for i,outchannel in enumerate(mlp2):
            new_state=conv2d('state%d'%i,new_state,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
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
def chamfer_big(pcd1, pcd2):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
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
#idx:batch*ptnum
#pointcloud:batch*16384*3
#output:batch*ptnum*3
def get_pts(pointcloud,idx):
    ptnum=pointcloud.get_shape()[1].value
    batchidx=tf.reshape(tf.range(tf.shape(pointcloud)[0]),[-1,ptnum,1])
    idmat=tf.concat([batchidx,tf.expand_dims(idx,axis=-1)],axis=-1)
    newptcloud=tf.gather_nd(pointcloud,idmat)
    return newptcloud
def groupin_near(ptmat):
    result=tf.reduce_sum(tf.square(ptmat),axis=-1)
    result=tf.reduce_mean(tf.reduce_mean(result,axis=-1),axis=-1)
    result=tf.reduce_mean(result)
    return result   
def zero_groupnear(ptcens,rawpts,outmat,name='balance_parameter'):
    _, _,dist,_ = tf_nndistance.nn_distance(ptcens, rawpts)
    inval=tf.reduce_mean(dist)
    outval=groupin_near(outmat)
    result=tf.nn.relu(outval-0.4*inval)
    return result
def get_gen(path,batch_size=BATCH_SIZE,input_ptnum=INNUM,output_ptnum=PTNUM,is_training=True):
    df, num = lmdb_dataflow(path,batch_size,input_ptnum,output_ptnum,is_training)
    gen = df.get_data()
    return gen,num
def train_one_batch(sess,ops,train_writer,batch):
    ids,batch_point,npts,output_point=next(ops['train_gen'])
    feed_dict = {ops['pointcloud_pl']: batch_point,ops['gt_pl']:output_point}
    resi = sess.run([ops['trainstep'],ops['loss'],ops['decfac'],ops['cd']], feed_dict=feed_dict)
    code1,code2,code3=sess.run([ops['code1'],ops['code2'],ops['code3']],feed_dict=feed_dict)
    code1=code1[0]
    code2=code2[0]
    code3=code3[0]
    if (batch+1) % 500 == 0:
        result = sess.run(ops['merged'], feed_dict=feed_dict)
        train_writer.add_summary(result, batch)
        print('epoch: %d'%(batch*BATCH_SIZE//ops['train_num']+1),'batch: %d' %batch)
        print('loss: ',resi[1])
        print('decode_cell regularization: ', resi[-1])
        print('max of code1 first: %f'%max(code1),'code1 nonzero num:%d'%len(code1[code1!=0]))
        print('max of code2 first: %f'%max(code2),'code2 nonzero num:%d'%len(code2[code2!=0]))
        print('max of code3 first: %f'%max(code3),'code3 nonzero num:%d'%len(code3[code3!=0]))

        print('loss: ',resi[1])
        print('emd loss',resi[2])
        print('cd loss',resi[3])
def eval_one_batch(sess,ops):
    chamferlist=[]
    emdlist=[]
    itertime=ops['valid_num']//EVAL_SIZE
    print('evaluate begin_________')
    for i in range(itertime):
        ids,batch_point,npts,output_point=next(ops['valid_gen'])
        feed_dict = {ops['pointcloud_pl']: batch_point,ops['gt_pl']:output_point}
        resi_chamfer,resi_emd = sess.run([ops['chamfer_loss'],ops['emd_loss']],feed_dict=feed_dict)
        chamferlist.append(resi_chamfer)
        emdlist.append(resi_emd)
    print('mean chamfer loss: %f'%mean(chamferlist))
    print('mean emd loss: %f'%mean(emdlist))
    print('evalueate end__________')
    return mean(emdlist),mean(chamferlist)

def train():   
    trainpath='../../dense_data/train.lmdb'
    valpath='../../dense_data/valid.lmdb'
    pointcloud_pl=tf.placeholder(tf.float32,[None,3000,3],name='pointcloud_pl')
    gt_pl=tf.placeholder(tf.float32,[None,PTNUM,3],name='gt_pl')
    
    out1,out2,out3,out4=full_process(pointcloud_pl)
    
    train_gen,train_num=get_gen(trainpath)
    valid_gen,valid_num=get_gen(valpath,batch_size=EVAL_SIZE,is_training=False)

    layernum1=out1.get_shape()[1].value
    layernum2=out2.get_shape()[1].value
    _,gt1=sampling(layernum1,gt_pl,use_type='f')
    _,gt2=sampling(layernum2,gt_pl,use_type='f')
    
    global_step=tf.Variable(0,trainable=False)
 
    alpha0 = tf.train.piecewise_constant(global_step, [50000,100000,150000,200000],
                                        [0.0005, 0.0002,0.0002,0.0001,0.00001], 'alpha_op')
    alpha1 = tf.train.piecewise_constant(global_step, [50000,150000],
                                        [0.01, 0.01,0.001], 'alpha_op') 

    chamfer_loss,ptid=chamfer_big(gt_pl,out4)
    emd_loss=earth_mover(gt_pl,out4)

    refinemove3=tf.get_collection('refine_layer_final16384')[0]
    moveloss=tf.reduce_mean(tf.reduce_sum(tf.square(refinemove3),axis=-1)) 
    cd1=earth_mover(gt1,tf.get_collection('points1')[0])
    cd2=earth_mover(gt2,tf.get_collection('points2')[0])
    cd3,_=chamfer_big(gt_pl,out3)
    cd4,_=chamfer_big(gt_pl,out4)
    recd3=re_chamfer(gt_pl,out3,part=8)
    
    loss=0.2*(cd1+cd2)+cd3+cd4+0.2*recd3+0.1*moveloss

    loss_d1=0.05*zero_groupnear(gt1,gt2,tf.get_collection('decode_cell'+str(layernum1))[0],name='balance_para1')
    loss_d2=0.05*zero_groupnear(gt2,gt_pl,tf.get_collection('decode_cell'+str(layernum2))[0],name='balance_para2')
    loss_dec=tf.add_n(tf.get_collection('decfactor'))
    loss=loss+loss_d1+loss_d2+alpha1*loss_dec

    var=[v for v in tf.global_variables()]

    trainstep=tf.train.AdamOptimizer(learning_rate=alpha0).minimize(loss,var_list=var, global_step=global_step)

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
             'loss':[cd1,cd2,loss_d1,loss_d2],
             'trainstep':trainstep,
             'out':out4,
             'merged':merged,
             'train_gen':train_gen,
             'valid_gen':valid_gen,
             'chamfer_loss':chamfer_loss,
             'emd_loss':emd_loss,
             'decfac':tf.get_collection('decfactor'),
             'cd':cd3+cd4,
             'train_num':train_num,
             'valid_num':valid_num,
             'code1':tf.squeeze(tf.get_collection('code1')[0]),
             'code2':tf.squeeze(tf.get_collection('code2')[0]),
             'code3':tf.squeeze(tf.get_collection('code3')[0]),
             }
        bestemd=100.0
        bestcd=100.0
        for i in range(BATCH_ITER_TIME):
            train_one_batch(sess,ops,writer,i)
            if (i+1)%20000==0:
                save_path = saver.save(sess, './modelvv_recon/model',global_step=i)
                meanemd,meancd=eval_one_batch(sess,ops)
                if meancd<bestcd:
                    bestemd=meanemd
                    bestcd=meancd
                    os.system('rm -r ./bestrecord')
                    os.system('mkdir ./bestrecord')
                    os.system('cp -r ./modelvv_recon ./bestrecord')
                    print('record bestsofar: ',bestemd,bestcd)
if __name__=='__main__':
    train()
