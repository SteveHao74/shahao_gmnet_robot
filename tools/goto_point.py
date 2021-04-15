'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2019-11-13 17:00:16
@LastEditTime: 2019-12-10 22:16:41
@LastEditors: Lai
'''
import sys
import os
import cv2
import numpy as np
file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(file_path, '..'))
print(root_path)
sys.path.append(root_path)
try:
    from utils.robotiq import Robotiq85
    from utils.ur_control import URSControl
    from utils.primesense_sensor import PrimesenseSensor
    from utils.transform import TransformAssist
except:
    print('import utils error')


class GotoTest(object):
    def __init__(self, camera, tran_path=None):
        self.camera = camera
        self.ur = URSControl()
        self.g = None
        self.start_point = None
        self.now_point = None
        self.is_move = False
        try:
            self.trans = TransformAssist.from_file(tran_path)
            self.ur = URSControl()
            self.grasp = Robotiq185()
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

    def ur_move(self, g, depth):
        p_image = g[:2]
        p_w = self.trans.pixel_in_world(depth, p_image)
        p_w[2] = 0.35
        angle = np.rad2deg(np.pi/2-g[2])
        print(angle)
        print(p_image)
        pose = self.ur.transform_to_base(p_w, angle=angle, show=False)
        print(pose)
        if self.ur.movej(pose, v=0.08):
            print('ur move sucess')
        else:
            print('ur move fail')

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
        plot_center(img, p0, axis, g[3]/2.5, width=3)
        plot_center(img, p1, axis, g[3]/2.5, width=3)
        cv2.circle(img, tuple(g[:2].astype('int')), 3, bule, -1)

    def main_loop(self):
        cv2.namedWindow('goto')
        cv2.setMouseCallback('goto', self.mouse_callback)
        while True:
            while True:
                ret, img, depth = self.camera.read(1)
                if ret:
                    break
                print('img read fail')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if self.g is not None:
                self.plot_grasp_cv(img, self.g)
            elif self.is_move:
                cv2.line(img, tuple(self.start_point.astype('int')),
                         tuple(self.now_point.astype('int')), [0, 0, 255], 2)
            cv2.imshow('goto', img)
            key = cv2.waitKey(20)
            if key == ord('q'):
                break
            elif key == ord('g') and self.g is not None:
                depth = depth[int(self.g[1]), int(self.g[0])]
                depth = depth + 0.05 if depth > 0.75 else 0.95
                self.ur_move(self.g, depth)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tran_path = os.path.join(root_path, 'transform')
    with PrimesenseSensor() as camera:
        go = GotoTest(camera, tran_path)
