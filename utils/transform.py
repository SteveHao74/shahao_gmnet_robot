'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2019-11-13 15:08:32
@LastEditTime: 2019-11-13 16:45:38
@LastEditors: Lai
'''
import os
import numpy as np
from scipy.spatial.transform import Rotation

x = Rotation.from_euler('x', 45, degrees=True).as_dcm()
z = Rotation.from_euler('z', 45, degrees=True).as_dcm()
BASE2WORLD = np.eye(4)
BASE2WORLD[:3, :3] = x.dot(z)
BASE2WORLD[:3, 3] = [0, -0.225, 0.4]
WORLD2BASE = np.linalg.inv(BASE2WORLD)


class TransformAssist():
    def __init__(self, cam2base, base2world=BASE2WORLD, cam_dist=None, cam_mtx=None):
        self.cam2base = cam2base
        self.base2world = base2world
        self.base2cam = np.linalg.inv(self.cam2base)
        self.world2base = np.linalg.inv(self.base2world)
        self.dist = cam_dist
        self.mtx = cam_mtx
        self.de_mtx = np.linalg.inv(self.mtx)

    @classmethod
    def from_file(cls, path, base2world=BASE2WORLD):
        cam2base = np.load(os.path.join(path, 'cam2base.npy'))
        cam_dist = np.load(os.path.join(path, 'dist.npy'))
        cam_mtx = np.load(os.path.join(path, 'mtx.npy'))
        return cls(cam2base, base2world, cam_dist, cam_mtx)

    def deproject_pixel(self, depth, pixel):
        point_3d = depth * self.de_mtx.dot(np.r_[pixel, 1.0])
        return point_3d

    def pixel_in_world(self, depth, pixel):
        point_cam = np.r_[self.deproject_pixel(depth, pixel), 1.0]
        point_base = self.cam2base.dot(point_cam)
        point_world = self.base2world.dot(point_base)
        return point_world[:3]
