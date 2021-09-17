import os
import h5py
import numpy as np
import math
import dataout
import copy
#from visualize import *
import random
from plyfile import PlyData
import scipy.io
from scipy import spatial
import lmdb
import pickle
import sys
sys.path.append('scannet')
import scannet_dataset as sc
from sklearn.cluster import KMeans
SAMPLE_NUM=100
SAMPLE_CEN=50
SAMPLE_R=0.4
FILE_NUM=1
BATCH_NUM=2

def load_h5(h5_filename):
    f=h5py.File(h5_filename)
    data=f['data'][:]
    return data
def load_h5label(h5_filename):
    f=h5py.File(h5_filename,'r')
    data=f['data'][:]
    label=f['label'][:]
    return data,label
def load_mat(matpath):
    f=scipy.io.loadmat(matpath)
    data=f['data'][:]
    label=f['label'][:]
    return data,label
def load_h5i(filename,n):
    f = h5py.File(filename)
    data = f['data'+str(n)][:]
    return data
def get_scannet_dataset(rootpath,npoints,split='train'):
    dataset=sc.ScannetDataset(root=rootpath,npoints=npoints,split=split)
    return dataset
def get_scannet_batch(dataset,npoints, start_idx, end_idx):
    idxs=np.arange(0,len(dataset))
    bsize = end_idx-start_idx
    result = np.zeros((bsize, npoints, 3))
    #batch_label = np.zeros((bsize, npoints), dtype=np.int32)
    #batch_smpw = np.zeros((bsize, npoints), dtype=np.float32)
    for i in range(bsize):
        ps,seg,smpw = dataset[idxs[i+start_idx]]
        #print(np.shape(ps))
        result[i,...] = ps
    dmax=np.max(result,axis=1,keepdims=True)
    dmin=np.min(result,axis=1,keepdims=True)
    length=(dmax-dmin)/2
    center=(dmax+dmin)/2
    result=(result-center)/length
        #batch_label[i,:] = seg
        #batch_smpw[i,:] = smpw
    return result,length,center
def get_scene_dataset(rootpath,npoints,split='train'):
    dataset=sc.ScannetDatasetWholeScene(root=rootpath,npoints=npoints,split=split)
    return dataset
def get_scene_batch(dataset,npoints,start_idx):
    ps,seg,smpw = dataset[start_idx]
    result = ps
    dmax=np.max(result,axis=1,keepdims=True)
    dmin=np.min(result,axis=1,keepdims=True)
    length=(dmax-dmin)/2
    center=(dmax+dmin)/2
    result=(result-center)/length
    #print(np.shape(result),np.shape(length),np.shape(center))
    return result,length,center
def get_scannet(rootpath,idx,split='train'):
    data_filename = os.path.join(rootpath, 'scannet_%s.pickle'%(split))
    with open(data_filename,'rb') as fp:
        points = pickle.load(fp,encoding='iso-8859-1')
        labels = pickle.load(fp,encoding='iso-8859-1')
    #print(np.shape(points[idx]))
    data=points[idx]
    return data
def get_batch_scannet(rootpath,idx,ptnum,split='train'):
    data=get_scannet(rootpath,idx,split='train')
    #print('>>>>>>>',np.shape(data))
    #random.shuffle(data)
    idx=np.random.choice(np.shape(data)[0],ptnum)
    #data=data[idx,:]
    datanum=np.shape(data)[0]
    spnum=int(datanum/ptnum)
    result=np.zeros((spnum,ptnum,3))
    for i in range(spnum):
        result[i]=data[i*ptnum:(i+1)*ptnum,:]
    dmax=np.max(result,axis=1,keepdims=True)
    dmin=np.min(result,axis=1,keepdims=True)
    length=(dmax-dmin)/2
    center=(dmax+dmin)/2
    result=(result-center)/length
    return result,length,center
def get_part_scannet(rootpath,idx,ptnum,split='train',centype='r'):
    data=get_scannet(rootpath,idx,split='train')
    datanum=np.shape(data)[0]
    cennum=int(datanum/ptnum)+1
    if centype is 'r':
        idx=np.random.choice(datanum,2048,replace=False)
        cens=data[idx]
        cens=KMeans(n_clusters=cennum, random_state=1).fit(cens)
        cens=cens.cluster_centers_
    result={}
    for i in range(datanum):
        dis=np.sum(np.square(data[i,:]-cens),axis=-1)
        minid=np.argmin(dis,axis=0)
        if minid not in result.keys():
            result[minid]=[data[i,:]]
        else:
            result[minid].append(data[i,:])
    data=[]
    for i in range(cennum):
        datai=result[i]
        #print(len(datai))
        idx=np.random.choice(len(datai),ptnum,replace=True)
        datai=np.expand_dims(np.array(datai)[idx,:],axis=0)
        data.append(datai)
    data=np.concatenate(data,axis=0)
    dmax=np.max(data,axis=1,keepdims=True)
    dmin=np.min(data,axis=1,keepdims=True)
    length=(dmax-dmin)/2
    center=(dmax+dmin)/2
    data=(data-center)/length
    print(np.shape(data),np.shape(center),np.shape(length))
    return data,length,center
def load_kitti(path):
    data=np.fromfile(path,dtype=np.float32,count=-1)
    data=np.reshape(data,[-1,4])[:,:3]
    #print(np.shape(data))
    return data
def load_kitti(base_path,sequence_id,filename):
    path=base_path+'/'+sequence_id+'/'+'velodyne/'+filename
    data=np.fromfile(path,dtype=np.float32,count=-1)
    data=np.reshape(data,[-1,4])[:,:3]
    return data
def load_k_kitti(base_path,sequence_id,filename,bsize=16,knum=2048):
    data=load_kitti(base_path,sequence_id,filename)
    random.shuffle(data)
    cens=data[:bsize,:]
    tree=spatial.KDTree(data)
    kdata=tree.query(cens,k=knum)
    result=data[kdata[1]]
    print(np.shape(result))
    #center=np.mean(result,axis=2,keepdims=True)#batch*1*3
    #stds=np.sqrt(np.var(result,axis=2,keepdims=True))
    dmax=np.max(result,axis=1,keepdims=True)
    dmin=np.min(result,axis=1,keepdims=True)
    length=(dmax-dmin)/2
    center=(dmax+dmin)/2
    result=(result-center)/length
    return result,length
def load_npz():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    train_npy = os.path.join(DATA_DIR, 'shapenet57448xyzonly.npz')
    td = dict(np.load(train_npy))
    data=td['data']
    return data
def getdir():
    BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR=os.path.join(BASE_DIR,'data')
    return DATA_DIR
def getdir2():
    BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR=os.path.join(BASE_DIR,'data2')
    return DATA_DIR

def getspdir():
    BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR=os.path.join(BASE_DIR,'data')
    DATA_DIR=os.path.join(DATA_DIR,'hdf5_data')
    return DATA_DIR
def getspdir2():
    BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR=os.path.join(BASE_DIR,'data2')
    DATA_DIR=os.path.join(DATA_DIR,'hdf5_data')
    return DATA_DIR
def getbigsp():
    BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR=os.path.join(BASE_DIR,'data')
    DATA_DIR=os.path.join(DATA_DIR,'shapeNet_data')
    return DATA_DIR
def getfile(path):
    return [line.strip('\n') for line in open(path)]
def data_normal(data):
    minone=np.min(data,axis=0)
    maxone=np.max(data,axis=0)
    cha=maxone-minone
    if(~ 0 in cha):
        data=(data-minone)/(cha)
    else:
        for i in range(3):
            if(cha[i]<0.001):
                cha[i]=cha[i]+1
        data = (data - minone) / (cha)
    return data,maxone,minone
def distance(a,b):
    dis=0
    #print(a,np.shape(a))
##    for i in range(len(a)):
##        dis=dis+pow(a[i]-b[i],2)
##    dis=math.sqrt(dis)
    
    return np.sqrt(np.sum(np.square(a-b)))
def minone(data,point):
    midis=1000
    ptargs=data[0]
    args=0
    for i in range(len(data)):
        pt=data[i,:]
        dis=distance(point,pt)
       # print(dis,midis)
        if dis<midis:
            midis=dis
            ptargs=pt
            args=i
    return midis,ptargs,args
#最远距离采样，确定中心点数
def far_sample(data,m,use_random=False):
    result=np.zeros((m,3))
    for i in range(m):
        maxdis=0
        if use_random:
            args=data[random.randint(0,len(data)-1)]
        else:
            args=data[0]
        for pt in data:
            dis,ptargs,_=minone(result,pt)
            if(dis>maxdis):
                maxdis=dis
                args=pt
        
        result[i,:]=args
        data=np.delete(data,args,axis=0)
        print(np.shape(data))
    return result
#获取最远距离采样中的周围区域，m是中心点数，n是包围球内的点数，r是包围球半径           
def ball_devide(data,m,n,r):
    batlen=len(data)
    pts_cnt=np.zeros((batlen,m))
    result=np.zeros((batlen,m,n,3))
    center=np.zeros((batlen,m,3))
    for l in range(batlen):
        center[l,:,:]=far_sample(data[l,:,:],m)
        for i in range(m):
            count=0
            for dt in data[l,:,:]:
                if count==n:
                    break
                if distance(dt,center[l,i,:])<r:
                    if count==0:
                        for k in range(n):
                            result[l,i,k,:]=dt
                    result[l,i,count,:]=dt
                    count=count+1
            pts_cnt[l,i]=count
    result=sort_by_dimension(result)
    return result,center,pts_cnt
def sort_by_dimension(data,dimen=0):
    result = np.zeros((data.shape[0],data.shape[1],data.shape[2], data.shape[3]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            order=np.argsort(data[i,j],dimen)
            for k in range(data.shape[2]):
                result[i,j,order[k],:]=data[i,j,k,:]
    return result

def normalize(data):
    partsize = np.shape(data)
    batchnum = partsize[0]
    censize = partsize[1]
    result = copy.deepcopy(data)
    maxones=np.zeros([batchnum,censize,3])
    minones=np.zeros([batchnum,censize,3])
    for b in range(batchnum):
        for c in range(censize):
            print(np.shape(data[b,c]))
            result[b, c],maxones[b,c],minones[b,c] = data_normal(data[b,c])
    return result,maxones,minones
def centerize(data,center):
    partsize = np.shape(data)
    batchnum = partsize[0]
    censize = partsize[1]
    result = copy.deepcopy(data)
    for b in range(batchnum):
        for c in range(censize):
            result[b, c] = data[b, c] - center[b,c]
    return result,center
def uncenterize(data,center):
    partsize = np.shape(data)
    batchnum = partsize[0]
    censize = partsize[1]
    result = copy.deepcopy(data)
    for b in range(batchnum):
        for c in range(censize):
            result[b, c] = data[b, c] + center[b,c]
    return result,center
def saveh5data(data,name,filename):
    f = h5py.File(filename, "w")
    f.create_dataset(name,data=data)
    f.close()
def create_traindata(data,m,n,r,name,filename):
    part, center, _ = ball_devide(data, m, n, r)
    traindata, _,_ = normalize(part)
    traindata = np.reshape(traindata, [-1, n, 3])
    saveh5data(traindata, name, filename)
def create_fromfile(trainfiles,batch_num,filenum):
    for i in range(filenum):
        data = load_h5(os.path.join(DATA_DIR, trainfiles[i]))
        create_traindata(data[0:batch_num],m,n,r,'data'+str(i),'train_data')
#聚类，data是数据，k是类数，返回每个模型被划分后的类别及其点（放在字典中）
def kmeans(data,k):
    dn,dm,ds=np.shape(data)
    kpoint=np.zeros((dn,k,ds))
    print(np.shape(kpoint))
    newdata={}
    for i in range(dn):
        c={}
        kpoint[i,:,:]=far_sample(data[i,:,:],k)
        for j in range(k):
            c[j]=kpoint[i,j,:]
        last=0.01
        kth=0
        lastime=1000
        temp=copy.deepcopy(c)
        while abs(last-lastime)/last>0.001 and lastime!=0:
            c=copy.deepcopy(temp)
            last=lastime
            group=np.zeros((k,ds))
            count=np.zeros(k)
            lastime=0
            for j in range(dm):
                dis,pt,kargs=minone(kpoint[i,:,:],data[i,j,:])
                c[kargs]=np.row_stack((c[kargs],data[i,j,:]))
                group[kargs,:]=group[kargs,:]+data[i,j,:]
                count[kargs]=count[kargs]+1
            for j in range(k):
                newcen=group[j,:]/count[j]
                lastime=lastime+distance(kpoint[i,j,:],newcen)
                kpoint[i,j,:]=newcen
            lastime=lastime/k
        newdata[i]=c
    return kpoint,newdata
def readcode(path):
    if os.path.isfile(path):
        data=[]
        f=open(path,'r')
        for line in f.readlines():
            strlist=line.split()
            data.append([float(i) for i in strlist])
        f.close()
        return data
    else:
        return -1
def add_zero(binstr,digitlen=32):
    binstr=binstr[2:]
    zerostr=''
    for i in range(digitlen-len(binstr)):
        zerostr=zerostr+'0'
    return zerostr+binstr
    
def dealdata(data,digitlen=32,devlen=8):
    length=len(data)
    result=np.zeros((length,int(digitlen*devlen)),dtype=np.float32)
    for i in range(length):
        code=data[i]
        serial_str=''
        if devlen>1:
            for j in range(int(devlen)):
                serial_str=serial_str+add_zero(bin(int(code[j])),digitlen)
        else:
            serial_str=add_zero(bin(int(code[0])),digitlen)
        serial_str=serial_str[-int(devlen*digitlen):]
        serial=[int(k) for k in serial_str]
        positions=np.array(serial).nonzero()[0]
        dev=int(devlen)
        if devlen<1:
            dev=1
        for k in range(len(code)-dev):
            result[i,positions[k]]=code[k+dev]
    return result
def void2array(data):
    result=[]
    for i in range(len(data)):
        result.append(list(data[i]))
    return np.array(result)
def data2ply(data,num,begin,path,file,ptnum=2048,char='w'):
    if not os.path.exists(path):
        os.makedirs(path)
    header='''ply
format ascii 1.0
element vertex '''+str(ptnum)+'''
property float x
property float y
property float z
end_header\n'''
    length=len(data)
    for i in range(num):
        posi=os.path.join(path,str(file)+'_'+str(i)+'.ply')
        f=open(posi,char)
        f.write(header)
        savestr=''
        for j in range(len(data[begin+i])):
            for k in range(3):
                datalist=data[i+begin,j]
                savestr=savestr+str(datalist[k])+' '
            savestr=savestr+'\n'
        f.write(savestr)
        f.close()
def load_ply(filename):
    plydata=PlyData.read(filename)
    data=void2array(plydata.elements[0].data)
    return data

def ply2data(filename):
    plydata=PlyData.read(filename)
    data=void2array(plydata.elements[0].data)
    if len(data)>2048:
        np.random.shuffle(data)
        data=data[:2048]
    elif len(data)<2048:
        return None,None
    return data,True
def plys2data(basepath,num):
    if not os.path.exists(basepath):
        return
    files=os.listdir(basepath)
    files.sort(key=lambda x:int(x[2:-4])+10000*int(x[0]))
    result=[]
    for i in range(num):
        posi=os.path.join(basepath,files[i])
        data,judge=ply2data(posi)
        if judge:
            result.append(data)
    result=np.array(result)        
    return result
def getfilelist(basepath):
    files=os.listdir(basepath)
    files.sort(key=lambda x:int(x[2:-4])+10000*int(x[0]))
    print(len(files))
    filenum=len(files)
    for i in range(filenum):
        files[i]=os.path.join(basepath,files[i])
    return files
def readplys(pathlist,start,end):
    result=[]
    for i in range(end-start+1):
        result.append(np.expand_dims(load_ply(pathlist[start+i]),axis=0))
    result=np.concatenate(result,axis=0)
    return result
def oc2data(basepath,path2):
    if not os.path.exists(basepath):
        return
    files=os.listdir(path2)
    files.sort(key=lambda x:int(x[2:-4])+10000*int(x[0]))
    
    result=[]
    for i in range(len(files)):
        posi=os.path.join(basepath,files[i])
        data,judge=ply2data(posi)
        if judge:
            result.append(data)
    result=np.array(result)
    return result

def test2plys(basepath,despath,filenames):
    for i in range(len(filenames)):
        data,label=load_h5label(os.path.join(basepath,filenames[i]))
        data2ply(data,len(data),0,despath,i)
def txt2ply(txtpath,plypath):
    data=np.loadtxt(txtpath)
    ptnum=len(data)
    header='''ply
format ascii 1.0
element vertex '''+str(ptnum)+'''
property float x
property float y
property float z
end_header\n'''
    f=open(plypath,'w')
    f.write(header)
    savestr=''
    for j in range(len(data)):
        for k in range(3):
            savestr=savestr+str(data[j,k])+' '
        savestr=savestr+'\n'
    f.write(savestr)
    f.close()    
if(__name__=="__main__"):
    #load_kitti('../../hard_disk/KITTI/odometry/sequences/00/velodyne/000000.bin') 
    #basepath='../../hard_disk/KITTI/odometry/sequences'
    #load_k_kitti(basepath,'00','000000.bin',bsize=16,knum=2048)
    data=get_scannet('../data',0,'train')
    dataout.display_mayavi(data)
    #env_db = lmdb.Environment('train') 
    # env_db = lmdb.open("./trainC") 
    #txn = env_db.begin() 
# get函数通过键值查询数据,如果要查询的键值没有对应数据，则输出None 
    #print(txn.get(str(200).encode()))
    #count=0 
    #for key, value in txn.cursor(): #遍历 
    #    count+=1
    #print(count)

    #env_db.close() 
    #txt2ply('D:/gpu_version_htxnet/data/txts/bunny2048.txt','D:/gpu_version_htxnet/data/txts/bunny2048.ply')
    
##    m=SAMPLE_CEN
##    n=SAMPLE_NUM
##    r=SAMPLE_R
##    filenum = FILE_NUM
##    DATA_DIR=getspdir()
##    filelist=os.listdir(DATA_DIR)
##    trainfiles=getfile(os.path.join(DATA_DIR,'train_files.txt'))
##    testfiles=getfile(os.path.join(DATA_DIR,'test_files.txt'))
##    data,label=load_h5label(os.path.join(DATA_DIR,trainfiles[0]))
####    dataout.display_mayavi(np.array(data[44]))
##    while(1):
##        print('we can input:')
##        ptid=int(input())
##        print(label[ptid])
##        dataout.display_mayavi(np.array(data[ptid]))

##    data=plys2data('thisplys',130)
##    data2=plys2data('octplys',130)
##    print(len(data),len(data2))
##    if len(data)>len(data2):
##        data=data[:len(data2)]
##    print(np.shape(data),np.shape(data2))
##    saveh5data(data2,'data','oct0.03_1.5972.h5')
##    saveh5data(data,'data','testdata03_1.h5')
##    print(np.shape(load_h5('pcl\oct0.03_1.5972.h5')))
##    print(np.shape(load_h5('testdata03.h5')))


##    data=ply2data('octplys/0_140.ply')
##    dataout.display(data)
##    data=load_h5('oct0.001_3.525.h5')
##    dataout.display(data[0])
##    test2plys(DATA_DIR,'testplys',testfiles)
##    n=0
##    data,label=load_h5label(os.path.join(DATA_DIR,trainfiles[n]))
##    data2ply(data,3,0,'plyfiles')
##    while(1):
##        print('we can input:')
##        ptid=int(input())
##        print(label[ptid])
##        fig = plt.figure(figsize=(12,4))
##        ax_a = plt.subplot(111, projection='3d')
##        ax_a.view_init(15,50)
##        draw_pts(data[ptid],None,'gray',ax=ax_a,sz=4)
##        plt.show()
    
##    data=load_npz()
##    datanum=np.shape(data)[0]
##    print(np.shape(data))
##    path=getbigsp()
##    trainfile_path=os.path.join(path,'train_files.txt')
##    f=open(trainfile_path,'w')
##    for i in range(1):
##        filename='sp_data_train'+str(i)+'.h5'
##        f.write(filename+'\n')
##        saveh5data(data[i*2048:(i+1)*2048],'data',os.path.join(path,filename))
##    f.close()
##    data=load_h5(os.path.join(path,'sp_data_train0.h5'))
##    print(np.shape(data))
##    dataout.display(data[3])
##    data,label=load_h5label(os.path.join(DATA_DIR,trainfiles[0]))
##    print(min(label),max(label))
##    for i in range(5):
##        print(np.shape(load_h5(os.path.join(DATA_DIR,testfiles[i]))))
##    dataout.display(data[7])
##    fig = plt.figure(figsize=(12,4))
##    ax_a = plt.subplot(111, projection='3d')
##    ax_a.view_init(15,50)
##    draw_pts(data[7],None,'gray',ax=ax_a,sz=5)
##    plt.show()
    # divdata,center,pts_cnt=ball_devide(np.array([data[0,:,:]]),m,n,r)
    # print(center)
    # dataout.display(data[0],center[0],np.reshape(divdata[0],[-1,3]))
    #
    # kpoint,newdata=kmeans(np.array([data[0]]),10)
    # c=newdata[0]
    # for i in range(10):
    #     print(np.shape(c[i]))
    # dataout.display(c[0],c[1],c[2],c[3],c[4])

    # create_fromfile(trainfiles,BATCH_NUM,FILE_NUM)

    # data1,center=centerize(divdata,center)
    # data2=uncenterize(data1,center)


    # divdata=load_h5i('train_data',0)
    # dataout.display(np.reshape(divdata[0],[-1,3]))



