# -*- coding: utf-8 -*-
'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-16 13:01:40
@LastEditTime : 2019-12-26 16:34:56
@LastEditors  : Lai
'''
import numpy as np


class Grasp2D(object):
    """
    2D夹爪类型，夹爪投影到深度图像上的坐标.
    这里使用的都是图像坐标和numpy数组的轴顺序相反
    """

    def __init__(self, center, angle, depth, width=0.0, z_depth=0.0, quality=None, coll=None):
        """ 一个带斜向因子z的2d抓取, 这里需要假定p1的深度比较大
        center : 夹爪中心坐标，像素坐标表示
        angle : 抓取方向和相机x坐标的夹角, 由深度较小的p0指向深度较大的p1, (-pi, pi)
        depth : 夹爪中心点的深度
        width : 夹爪的宽度像素坐标
        z_depth: 抓取端点到抓取中心点z轴的距离,单位为m而非像素
        quality: 抓取质量
        coll: 抓取是否碰撞
        """
        self.center = center
        self.angle = angle
        self.depth = depth
        self.width_px = width
        self.z_depth = z_depth
        self.quality = quality
        self.coll = coll

    @property
    def axis(self):
        """ Returns the grasp axis. """
        return np.array([np.cos(self.angle), np.sin(self.angle)])

    @property
    def endpoints(self):
        """ Returns the grasp endpoints """
        p0 = self.center - (float(self.width_px) / 2) * self.axis
        p1 = self.center + (float(self.width_px) / 2) * self.axis
        p0 = p0.astype(np.int)
        p1 = p1.astype(np.int)
        d0 = self.depth - self.z_depth
        d1 = self.depth + self.z_depth
        return p0, p1, d0, d1

    def zoom(self, scale):
        """ 缩放图像时抓取变化为新的, 深度和图像无关所以不用变 """
        center = self.center * scale
        width = self.width_px * scale
        g = self.copy()
        g.width_px = width
        g.center = center
        return g

    def crop(self, crop_start):
        """ 裁剪图像时抓取的变化 """
        g = self.copy()
        g.center = g.center - crop_start
        return g

    @staticmethod
    def image_dist(g1, g2, alpha=1.0):
        """ 计算两个抓取在像素坐标下的距离
        """
        # point to point distances
        point_dist = np.linalg.norm(g1.center - g2.center)

        # axis distances
        dot = np.abs(g1.axis.dot(g2.axis))
        if dot > 1:
            dot = 1
        axis_dist = np.arccos(dot)

        return point_dist + alpha * axis_dist

    def to_saver(self):
        """ 保存成一个数组 """
        s = np.zeros((8,))
        s[:2] = self.center
        s[2] = self.depth
        s[3] = self.angle
        s[4] = self.width_px
        s[5] = self.z_depth
        s[6] = self.quality if self.quality is not None else -1
        s[7] = self.coll if self.coll is not None else -1
        return s

    @classmethod
    def from_saver(cls, s):
        """ 从数组中生成一个抓取 """
        center = s[:2]
        depth = s[2]
        angle = s[3]
        width = s[4]
        z_depth = s[5]
        if len(s) > 6:
            qualit = s[6] if s[6] != -1 else None
            coll = s[7] if s[7] != -1 else None
        return cls(center, angle, depth, width, z_depth, qualit, coll)

    def copy(self):
        return type(self)(self.center, self.angle, self.depth, self.width_px, self.z_depth, self.quality, self.coll)

    @classmethod
    def from_endpoint(cls, p0, p1, d0=0, d1=0, quality=None, coll=None):
        """ 通过两个末端点生成一个抓取
        p0, p1 : 末端点的像素坐标
        d0, d1 : p0, p1对应的深度
        这里的点也是图像坐标下的点x,y的顺序，而不是numpy坐标
        """
        p0, p1 = np.array(p0), np.array(p1)
        if d1 < d0:
            d0, d1 = d1, d0
            p0, p1 = p1, p0
        depth = (d0 + d1) / 2
        z_depth = (d1 - d0) / 2
        width = np.linalg.norm(p1 - p0)
        center_px = (p0 + p1) / 2
        axis = p1 - p0
        if np.linalg.norm(axis) > 0:
            axis = axis / np.linalg.norm(axis)
        if axis[1] > 0:
            angle = np.arccos(axis[0])
        else:
            angle = -np.arccos(axis[0])
        return cls(center_px, angle, depth, width, z_depth, quality, coll)
