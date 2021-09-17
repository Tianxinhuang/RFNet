# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np

#def plot_pcd_three_views(filename, pcds, titles, suptitle='', sizes=None, cmap='inferno', zdir='y',
#                         xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
#    #print(suptitle)
#    if sizes is None:
#        sizes = [0.5 for i in range(len(pcds))]
#    fig = plt.figure(figsize=(len(pcds) * 3, 9))
#    for i in range(3):
#        elev = 30
#        azim = -45 + 90 * i
#        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
#            #if colorlist is None:
#            color = pcd[:, 0]
#            #else:
#            #    color = colorlist[i]
#            ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection='3d')
#            ax.view_init(elev, azim)
#            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
#            ax.set_title(titles[j])
#            ax.set_axis_off()
#            ax.set_xlim(xlim)
#            ax.set_ylim(ylim)
#            ax.set_zlim(zlim)
#    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
#    plt.suptitle(suptitle)
#    fig.savefig(filename)
#    plt.close(fig)
def plot_pcd_three_views(filename, pcds, titles, suptitle='', sizes=None, cmap='inferno', zdir='y',
                         xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    #if sizes is None:
    sizes = [5 for i in range(len(pcds))]
    #fig = plt.figure(figsize=(len(pcds) * 3, 9))
    folder=filename.split('.')[0]
    print(filename.split('.')[0])
    #os.system('rm -r '+ folder)
    #os.system('mkdir '+ folder)
    import os
    os.makedirs(folder, exist_ok=True)
    for i in range(3):

        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            fig=plt.figure()
            filename=os.path.join(folder,'%s_%s.png'%(titles[j],str(i)))
            color = pcd[:, 0]
            #left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
            ax = fig.add_subplot(projection='3d')
            #ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection='3d')
            #ax=fig.add_axes()
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            #ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
            plt.suptitle(suptitle)
            fig.savefig(filename)
            plt.close(fig)
def plot_pcd_atten_views(filename, pcds, titles, colorlist=None, sizes=None, cmap='inferno', zdir='y',
                         xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    #if sizes is None:
    #sizes = np.array([5 for i in range(len(pcds))])
    #idx=(colorlist==-1.0)
    #print(idx)
    #sizes[idx]=15
    #assert False
    #fig = plt.figure(figsize=(len(pcds) * 3, 9))
    folder=filename.split('.')[0]
    print(filename.split('.')[0])
    #os.system('rm -r '+ folder)
    #os.system('mkdir '+ folder)
    import os
    os.makedirs(folder, exist_ok=True)
    for i in range(3):

        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            if colorlist is None:
                color = pcd[:, 0]
            else:
                color = colorlist[j]
            idx=(color==-1.0)
            sizes=np.ones_like(color)*20
            #print(idx)
            #print(np.shape(color),np.shape(sizes))
            sizes[idx]=50
            #assert False
            fig=plt.figure()
            filename=os.path.join(folder,'%s_%s.png'%(titles[j],str(i)))
            #color = pcd[:, 0]
            #left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
            ax = fig.add_subplot(projection='3d')
            #ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection='3d')
            #ax=fig.add_axes()
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=sizes, cmap=cmap, vmin=-1.0, vmax=0.5,alpha=0.5)
            #print('........',np.shape(pcd[idx,0]))
            ax.scatter(pcd[idx, 0], pcd[idx, 1], pcd[idx, 2], zdir=zdir, c=-1*np.ones_like(pcd[idx,0]), s=50, cmap=cmap, vmin=-1.0, vmax=0.5,alpha=1)
            #ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
            #plt.suptitle(suptitle)
            fig.savefig(filename)
            plt.close(fig)
#def plot_pcd_atten_views(filename, pcds, titles,colorlist=None, sizes=None, cmap='inferno', zdir='y',
#                         xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
#    if sizes is None:
#        sizes = [0.5 for i in range(len(pcds))]
#    fig = plt.figure(figsize=(len(pcds) * 3, 9))
#    for i in range(3):
#        elev = 30
#        azim = -45 + 90 * i
#        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
#            if colorlist is None:
#                color = pcd[:, 0]
#            else:
#                color = colorlist[j]
#            ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection='3d')
#            ax.view_init(elev, azim)
#            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
#            ax.set_title(titles[j])
#            ax.set_axis_off()
#            ax.set_xlim(xlim)
#            ax.set_ylim(ylim)
#            ax.set_zlim(zlim)
#    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
#    #plt.suptitle(suptitle)
#    fig.savefig(filename)
#    plt.close(fig)
