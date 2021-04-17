'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2019-11-13 17:00:16
@LastEditTime : 2019-12-25 20:03:45
@LastEditors  : Lai
'''
import sys
import time
import os
import cv2
import Pyro4
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import numpy as np
import matplotlib.pyplot as plt
file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(file_path, '..'))
print(root_path)
sys.path.append(root_path)
gmnet_path = os.path.abspath(os.path.join(root_path, '../gmnet/tools'))
sys.path.append(gmnet_path)
try:
    from utils.robotiq import Robotiq85
    # from utils.ur_control import URSControl
    from utils.primesense_sensor import PrimesenseSensor
    from utils.transform import TransformAssist
    from utils.image_preprocess import ImageProcessor
except:
    print('import utils error')
Pyro4.config.SERIALIZER = 'pickle'
grasp_server = Pyro4.Proxy("PYRO:grasp@10.12.218.239:6666")

UR_HOST = '192.168.1.101'


def planer(depth):
    print(depth.shape)
    im = ImageProcessor(depth)
    im300 = im.process
    np.save('aaa', im300)
    print(im300.shape)
    for i in range(5):
        try:
            g = grasp_server.plan(im300)
        except Exception:
            print("崩了一次")
        else:
            break
    print(g)
    center = im.process_to_original(g[:2])
    g = np.r_[center, -g[2:], 80]
    print(g)
    # g=np.array([1,1,1,1])
    return g


class GotoTest(object):
    def __init__(self, camera, tran_path=None):
        self.camera = camera
        self.g = None
        self.start_point = None
        self.now_point = None
        self.is_move = False
        self.is_planing = False
        self.depth = None
        try:
            self.trans = TransformAssist.from_file(tran_path)
            self.urc = RTDEControlInterface(UR_HOST)
            self.urs = RTDEReceiveInterface(UR_HOST)
            self.gripper = Robotiq85()
        except:
            print('ur connect fail or not transform matrix')
        self.main_loop()

    def mouse_callback(self, event, x, y, flags, param):
        # 按下左键的时候记录起始坐标
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = np.array([x, y])
            self.g = None
        elif event == cv2.EVENT_LBUTTONUP:
            end_point = np.array([x, y])
            # 计算得到一个抓取
            if self.is_planing:
                self.g = self.plan(self.depth, self.start_point, end_point)
                self.is_planing = False
            else:
                self.g = self.get_grasp(self.start_point, end_point)
            self.is_move = False
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            self.now_point = np.array([x, y])
            self.is_move = True

    def get_grasp(self, start, end, width=30):
        """ 通过起始点和终点，计算图像坐标上的抓取 """
        center = (start + end) / 2
        axis = end - start
        if (axis == 0).all():
            return None
        axis = axis / np.linalg.norm(axis)
        axis = -axis if axis[0] < 0 else axis
        angle = np.arcsin(np.clip(axis[1], -1, 1))
        # angle = angle if axis[1] > 0 else np.pi - angle
        return np.r_[center, angle, width]

    def get_real_width(self, g, depth):
        g_axis = np.array([np.cos(g[2]), np.sin(g[2])])
        p0, p1 = g[:2] - g_axis * g[3] * 0.5, g[:2] + g_axis * g[3] * 0.5
        p0_w = self.trans.pixel_in_world(depth, p0)
        p1_w = self.trans.pixel_in_world(depth, p1)
        width = np.linalg.norm((p0_w-p1_w)[:2])
        return width

    def g_to_world(self, g, depth):
        p_image = g[:2]
        p_w = self.trans.pixel_in_world(depth, p_image)
        return p_w, g[2]

    def goto_pose(self, pose, v):
        def compare(a, b, error=0.001):
            e = np.linalg.norm(np.array(a) - np.array(b))
            return e <= error
        self.urc.moveL(pose, speed=v)
        while not compare(self.urs.getTargetTCPPose(), pose, 0.001):
            time.sleep(0.1)

    def pick_it(self, g, depth):
        self.gripper.open()
        p_w, angle = self.g_to_world(g)
        p_w1 = np.r_[p_w[:2],  0.2]
        pose1 = self.ur.transform_to_base(p_w1, angle=angle, show=False)
        self.goto_pose(pose1, 0.25)
        p_w2 = np.r_[p_w[:2],  0.03]
        pose2 = self.ur.transform_to_base(p_w2, angle=angle, show=False)
        self.goto_pose(pose2, 0.15)
        self.goto_pose(pose1, 0.15)
        pose3 = np.array([0.4, -0.25, 0.2])
        self.goto_pose(pose3, 0.15)
        self.gripper.open()

    def ur_move(self, g, depth):
        self.pick_it(g, depth)

    def plot_grasp_cv(self, img, g, offset=[0, 0]):
        """ 使用opencv在图像上展示一个夹爪 """
        red = (0, 0, 255)
        bule = (255, 0, 0)

        def plot_2p(img, p0, p1, color=red, width=2):
            cv2.line(img, tuple(p0.astype('int')), tuple(p1.astype('int')), color, width)

        def plot_center(img, center, axis, length, color=red, width=2):
            axis = axis / np.linalg.norm(axis)
            p0 = center - axis * length / 2
            p1 = center + axis * length / 2
            plot_2p(img, p0, p1, color, width)

        g_axis = np.array([np.cos(g[2]), np.sin(g[2])])
        p0, p1 = g[:2] - g_axis * g[3] * 0.5, g[:2] + g_axis * g[3] * 0.5
        axis = [g_axis[1], -g_axis[0]]
        plot_2p(img, p0, p1)
        plot_center(img, p0, axis, g[3]/4.5, width=3)
        plot_center(img, p1, axis, g[3]/4.5, width=3)
        cv2.circle(img, tuple(g[:2].astype('int')), 3, bule, -1)

    def plan(self, depth, start, end):
        # if (np.abs(start - end) < 50).any():
        #     print('too small')
        #     return None
        # else:
        #     print(start, end)
        # x_min, x_max = min(start[0], end[0]), max(start[0], end[0])
        # y_min, y_max = min(start[1], end[1]), max(start[1], end[1])
        g = planer(depth)
        return g

    def read_image(self):
        while True:
            ret, img, depth = self.camera.read(1)
            if ret:
                break
            print('img read fail')
        # img = cv2.imread(os.path.join(root_path, 'test_images/1.jpg'))
        # depth = np.load(os.path.join(root_path, 'test_images/1.npy'))
        return img, depth

    def main_loop(self):
        cv2.namedWindow('goto')
        cv2.setMouseCallback('goto', self.mouse_callback)
        while True:
            img, self.depth = self.read_image()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if self.g is not None:
                self.plot_grasp_cv(img, self.g)
            elif self.is_move:
                if self.is_planing:
                    cv2.rectangle(img, tuple(self.start_point.astype('int')),
                                  tuple(self.now_point.astype('int')), [0, 0, 255])
                else:
                    cv2.line(img, tuple(self.start_point.astype('int')),
                             tuple(self.now_point.astype('int')), [0, 0, 255], 2)
            cv2.imshow('goto', img)
            key = cv2.waitKey(50)
            if key == ord('q'):
                break
            elif key == ord('g') and self.g is not None:
                depth = self.depth[int(self.g[1]), int(self.g[0])]
                depth = depth + 0.05 if depth > 0.75 else 0.95
                depth = 0.86
                self.ur_move(self.g, depth)
            elif key == ord('p'):
                self.is_planing = True
                self.g = None
            elif key == ord('r'):
                self.g = self.plan(self.depth, None, None)
                # self.g = self.plan(depth, rgb)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tran_path = os.path.join(root_path, 'transform')
    with PrimesenseSensor() as camera:
        go = GotoTest(camera, tran_path)
