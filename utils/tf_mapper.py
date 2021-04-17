'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2020-01-04 14:50:13
@LastEditTime : 2020-01-17 14:04:59
@LastEditors  : Lai
'''
from cv2 import cv2
import numpy as np
from scipy.spatial.transform import Rotation
height_min = 0.143#一定要高于这个高度，避免爪子碰到

def np2ur(pose):
    rotv = Rotation.from_dcm(pose[:3, :3]).as_rotvec()
    return np.r_[pose[:3, 3], rotv]

def  vector_to_matrix(vector):
    T_matrix = np.eye(4)
    r_vector = vector[3:]
    t_vector = vector[:3]
    R_matrix,_ = cv2.Rodrigues(r_vector)
    T_matrix[:3, :3] = R_matrix
    T_matrix[0, 3] = t_vector[0] 
    T_matrix[1,3] = t_vector[1]
    T_matrix[2,3] = t_vector[2]
    return T_matrix

class TFMapper_EyeInHand(object):
    """ 下面定义一些坐标系的缩写
    c: 相机坐标系
    b: 基坐标系
    w: 世界坐标系
    g: 夹爪坐标系
    """
    # x = Rotation.from_euler('x', 45, degrees=True).as_dcm()
    # z = Rotation.from_euler('z', 45, degrees=True).as_dcm()
    # T_b_w = np.eye(4)
    # T_b_w[:3, :3] = x.dot(z)
    # T_b_w[:3, 3] = [0, -0.225, 0.4]
    # T_w_b = np.linalg.inv(T_b_w)
    #以上是针对基座相对于桌面歪着的机械臂时使用的，这时需要仔细考虑世界坐标系和基坐标系之间的关系，而
    #而以下是 针对 机械臂基座 水平落放在桌面上时，则世界坐标系原则和基坐标系重合即可
    T_b_w  = np.eye(4)
    T_w_b  = np.eye(4)
    

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

    def grasp2d_to_matrix(self, g, T_g_b, h=height_min, tcp='camera'):#问题出在这
        """ 直接从相机坐标系中的抓取得到一个变换矩阵,可以用作相机也可以用作爪子
        需要主要的是，相机和爪子坐标系的y轴相反,且正负都可,需要执行时候确定
        """
        if tcp == 'camera':
            T_t_w = self.T_b_w.dot(T_g_b).dot(self.T_c_g)
        else:
            T_t_w = self.T_b_w.dot(T_g_b)
        p0, p1 = self.grasp2d_to_3d(g, T_g_b)
        matrix = self.grasp_to_matrix_pro(p0, p1,(p0+p1)/2, T_t_w)
        matrix = self.matrix_translation(matrix, z=h)
        return matrix, np.linalg.norm(p0-p1)

    def grasp_to_matrix_pro(self, p0, p1, center, T_t_w):
        """ 找到与当前状态最接近的那个抓取位姿,因为抓取轴正反都可以 """
        m0 = self.grasp_to_matrix(p0, p1,center)#得到了target相对于world的齐次变换阵
        # xy轴都相反z轴不变
        m1 = m0.copy()
        m1[:3, :2] = -m1[:3, :2]#把X，Y轴都取反
        r0 = np.linalg.inv(m0[:3, :3]).dot(T_t_w[:3, :3])
        c0 = np.sum(np.abs(Rotation.from_dcm(r0).as_euler('xyz')))
        r1 = np.linalg.inv(m1[:3, :3]).dot(T_t_w[:3, :3])
        c1 = np.sum(np.abs(Rotation.from_dcm(r1).as_euler('xyz')))
        return m0 if c0 < c1 else m1

    @staticmethod
    def grasp_to_matrix(p0, p1,center):
        """ 计算在抓取路径, 即与z轴最近的方向,这个是在物体坐标系下计算 """
        # 这里的y轴正反方向不确定,需要执行的时候再确定
        x = p0 - p1
        x = x / np.linalg.norm(x)#
        # z轴就是世界坐标系的z轴在抓取轴为法线的平面上的投影
        z_world = np.array([0, 0, -1])
        z = z_world - x * (x.dot(z_world))
        z = z / np.linalg.norm(z)
        y = np.cross(z, x)
        T_t_w = np.eye(4)
        T_t_w[:3, :3] = np.c_[x, y, z]#抓取目标姿态，以规划方向为x轴，与夹爪本身开和方向为X轴是一致的
        T_t_w[:3, 3] = center#这是抓取中心点的坐标，作为平移向量
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
        self.dis_cam_to_table=abs(T_c_b[2,3]) 
        print("相机到桌面的高度",self.dis_cam_to_table)

    def grasp2d_to_3d(self, g, grasp_depth):
        p0, p1,d0, d1 = g.endpoints
        d0 = grasp_depth #这里作了弊，直接给了物体最高高度向下一半夹爪的深度
        d1 = grasp_depth
        center_2 = (p0+p1)/2
        # 得到相机坐标系下的末端点
        p0_3d = self.camera.deproject(p0, d0)#这里出问题！！！！！,从服务器返回的深度是0
        p1_3d = self.camera.deproject(p1, d1)
        center_cam = self.camera.deproject(center_2, grasp_depth)
        # 转换到世界坐标系 
        center_base = self.T_c_b.dot(np.r_[center_cam,1])
        center_world = self.T_b_w.dot(center_base)[:3]
        print("两点在相机坐标系下",p0_3d,p1_3d)
        p0_base = self.T_c_b.dot(np.r_[p0_3d, 1])
        p1_base = self.T_c_b.dot(np.r_[p1_3d,1])
        print("两点在基座坐标系下",p0_base, p1_base)
        p0_world = self.T_b_w.dot(p0_base)[:3]
        p1_world = self.T_b_w.dot(p1_base)[:3]
        return p0_world, p1_world,center_world

    def grasp2d_to_matrix(self, g, T_g_b, grasp_depth):#这里出问题！！！！！
        """ 直接从相机坐标系中的抓取得到一个变换矩阵,可以用作相机也可以用作爪子
        需要主要的是，相机和爪子坐标系的y轴相反,且正负都可,需要执行时候确定
        """
        T_t_w = self.T_b_w.dot(T_g_b)#得到夹抓关于世界坐标系的位姿
        print("matir_0",T_t_w)
        p0, p1,center  = self.grasp2d_to_3d(g,grasp_depth)#得到两个抓取点在世界坐标系下的位置
        print('----------------p0', p0)
        print('----------------p1', p1)
        # p0[2] = (p0[2]+p1[2]) /2
        # # p1[2] = (p0[2]+p1[2]) /2
        p0[2] = center[2]#self.dis_cam_to_table-grasp_depth
        p1[2] = center[2]#self.dis_cam_to_table-grasp_depth
        print(center)
        print('----------------p0', p0)
        print('----------------p1', p1)
        matrix = self.grasp_to_matrix_pro(p0, p1, center, T_t_w)#得到了target相对于world的齐次变换阵
        print("matir_1",matrix)
        # matrix = self.matrix_translation(matrix, z=h)
        print("matir_2",matrix)
        return matrix, np.linalg.norm(p0-p1)

