
def readfile(path):
    with open(path,'r') as f:
        datalist=f.readlines()
        fidelist=[]
        mmdlist=[]
        for i,data in enumerate(datalist):
            strlist=data.split(': ')
            if strlist[0]=='fidelity':
                #if i<100:
                fidelist.append(float(strlist[1].replace('\n','')))
            elif strlist[0]=='MMD':
                #if i<1300:
                mmdlist.append(float(strlist[1].replace('\n','')))
            #print(strlist)
            #assert False
    print('mean fidelity: ',int(1e4*sum(fidelist)/len(fidelist))/1e4)
    print('mean MMD: ',int(1e4*sum(mmdlist)/len(mmdlist))/1e4)
if __name__=='__main__':
    readfile('kitti_result.cp')
    #readfile('kitti_result.txt')
