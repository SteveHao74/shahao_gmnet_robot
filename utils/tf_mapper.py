'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2020-01-04 14:50:13
@LastEditTime : 2020-01-17 14:04:59
@LastEditors  : Lai
'''
import numpy as np
from scipy.spatial.transform import Rotation


def np2ur(pose):
    rotv = Rotation.from_dcm(pose[:3, :3]).as_rotvec()
    return np.r_[pose[:3, 3], rotv]


class TFMapper_EyeInHand(object):
    """ 下面定义一些坐标系的缩写
    c: 相机坐标系
    b: 基坐标系
    w: 世界坐标系
    g: 夹爪坐标系
    """
    x = Rotation.from_euler('x', 45, degrees=True).as_dcm()
    z = Rotation.from_euler('z', 45, degrees=True).as_dcm()
    T_b_w = np.eye(4)
    T_b_w[:3, :3] = x.dot(z)
    T_b_w[:3, 3] = [0, -0.225, 0.4]
    T_w_b = np.linalg.inv(T_b_w)

    def __init__(self, camera, T_c_g):
        """ camera: 相机对象,实现grasp_2d到grasp_3d的变换
        T_c_g: 相机坐标系到工具坐标系的变换,手眼标定得到
        """
        self.camera = camera
        self.T_c_g = T_c_g

    def grasp2d_to_3d(self, g, T_g_b):
        p0, p1, d0, d1 = g.endpoints
        # 得到相机坐标系下的末端点
        p0_3d = self.camera.deproject(p0, d0)
        p1_3d = self.camera.deproject(p1, d1)
        # 转换到世界坐标系
        p0_world = self.camera_to_world(p0_3d, T_g_b)
        p1_world = self.camera_to_world(p1_3d, T_g_b)
        return p0_world, p1_world

    def camera_to_world(self, p, T_g_b):
        p_grasp = self.T_c_g.dot(np.r_[p, 1])
        p_base = T_g_b.dot(p_grasp)
        p_world = self.T_b_w.dot(p_base)
        return p_world[:3]

    def grasp_from_camera(self, T_c_w):
        """ 相机坐标系到世界坐标的变换得到tcp坐标到基座坐标的变换 """
        T_g_w = T_c_w.dot(np.linalg.inv(self.T_c_g))
        T_g_b = self.T_w_b.dot(T_g_w)
        return T_g_b

    def base_from_world(self, T_g_w):
        """ 相机坐标系到世界坐标的变换得到tcp坐标到基座坐标的变换 """
        T_g_b = self.T_w_b.dot(T_g_w)
        return T_g_b

    def grasp2d_to_matrix(self, g, T_g_b, h=-0.235, tcp='camera'):
        """ 直接从相机坐标系中的抓取得到一个变换矩阵,可以用作相机也可以用作爪子
        需要主要的是，相机和爪子坐标系的y轴相反,且正负都可,需要执行时候确定
        """
        if tcp == 'camera':
            T_t_w = self.T_b_w.dot(T_g_b).dot(self.T_c_g)
        else:
            T_t_w = self.T_b_w.dot(T_g_b)
        p0, p1 = self.grasp2d_to_3d(g, T_g_b)
        matrix = self.grasp_to_matrix_pro(p0, p1, T_t_w)
        matrix = self.matrix_translation(matrix, z=h)
        return matrix, np.linalg.norm(p0-p1)

    def grasp_to_matrix_pro(self, p0, p1, T_t_w):
        """ 找到与当前状态最接近的那个抓取位姿,因为抓取轴正反都可以 """
        m0 = self.grasp_to_matrix(p0, p1)
        # xy轴都相反z轴不变
        m1 = m0.copy()
        m1[:3, :2] = -m1[:3, :2]
        r0 = np.linalg.inv(m0[:3, :3]).dot(T_t_w[:3, :3])
        c0 = np.sum(np.abs(Rotation.from_dcm(r0).as_euler('xyz')))
        r1 = np.linalg.inv(m0[:3, :3]).dot(T_t_w[:3, :3])
        c1 = np.sum(np.abs(Rotation.from_dcm(r1).as_euler('xyz')))
        return m0 if c0 < c1 else m1

    @staticmethod
    def grasp_to_matrix(p0, p1):
        """ 计算在抓取路径, 即与z轴最近的方向,这个是在物体坐标系下计算 """
        # 这里的y轴正反方向不确定,需要执行的时候再确定
        x = p0 - p1
        x = x / np.linalg.norm(x)
        # z轴就是世界坐标系的z轴在抓取轴为法线的平面上的投影
        z_world = np.array([0, 0, -1])
        z = z_world - x * (x.dot(z_world))
        z = z / np.linalg.norm(z)
        y = np.cross(z, x)
        T_t_w = np.eye(4)
        T_t_w[:3, :3] = np.c_[x, y, z]
        T_t_w[:3, 3] = (p0 + p1) / 2
        return T_t_w

    @staticmethod
    def matrix_translation(T, x=0, y=0, z=0):
        # 坐标系沿着自己的轴平移
        Tt = np.eye(4)
        Tt[:3, 3] = [x, y, z]
        return T.dot(Tt)


class TFMapper_EyeToHand(TFMapper_EyeInHand):

    def __init__(self, camera, T_c_b):
        """ camera: 相机对象,实现grasp_2d到grasp_3d的变换
        T_c_b: 相机坐标系到base坐标系的变换,手眼标定得到
        """
        self.camera = camera
        self.T_c_b = T_c_b

    def grasp2d_to_3d(self, g):
        p0, p1, d0, d1 = g.endpoints
        # 得到相机坐标系下的末端点
        p0_3d = self.camera.deproject(p0, d0)
        p1_3d = self.camera.deproject(p1, d1)
        # 转换到世界坐标系
        p0_base = self.T_c_b.dot(np.r_[p0_3d, 1])
        p1_base = self.T_c_b.dot(np.r_[p1_3d,1])
        p0_world = self.T_b_w.dot(p0_base)[:3]
        p1_world = self.T_b_w.dot(p1_base)[:3]
        return p0_world, p1_world

    def grasp2d_to_matrix(self, g, T_g_b, h=-0.235):
        """ 直接从相机坐标系中的抓取得到一个变换矩阵,可以用作相机也可以用作爪子
        需要主要的是，相机和爪子坐标系的y轴相反,且正负都可,需要执行时候确定
        """
        T_t_w = self.T_b_w.dot(T_g_b)
        p0, p1 = self.grasp2d_to_3d(g)
        print('----------------p0', p0)
        print('----------------p1', p1)
        # p0[2] = (p0[2]+p1[2]) /2
        # p1[2] = (p0[2]+p1[2]) /2
        p0[2] = -0.02
        p1[2] = -0.02
        matrix = self.grasp_to_matrix_pro(p0, p1, T_t_w)
        matrix = self.matrix_translation(matrix, z=h)
        return matrix, np.linalg.norm(p0-p1)
