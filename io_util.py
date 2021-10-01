# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import numpy as np
from open3d import *
import open3d as o3d

def read_pcd(filename):
    pcd = io.read_point_cloud(filename)
    return np.array(pcd.points)


def save_pcd(filename, points):
    pcd = geometry.PointCloud()
    pcd.points = utility.Vector3dVector(points)
    io.write_point_cloud(filename, pcd)
