'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2020-01-04 14:35:01
@LastEditTime : 2020-01-17 17:28:56
@LastEditors  : Lai
'''

import os
import time
import sys
import shutil
import cv2
from glob import glob
import Pyro4
import numpy as np
import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg   # NavigationToolbar2TkAgg
file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(file_path, '..'))
print(root_path)
sys.path.append(root_path)
try:
    # from utils.primesense_sensor import PrimesenseSensor
    from utils.realsenseL515 import RealsenseSensor
    from utils.image_preprocess import ImageProcessor
    from utils.grasp_2d import Grasp2D
    from utils.tf_mapper import TFMapper_EyeInHand
    from utils.tf_mapper import np2ur
    from utils.kinematics1 import ur2np
    from utils.ur_servo import URServo
    from utils.robotiq_85 import Robotiq85
except Exception as e:
    print('import utils error')
    raise e


Pyro4.config.SERIALIZER = 'pickle'
grasp_server = Pyro4.Proxy("PYRO:Planer@10.12.218.249:6666")
Pyro4.asyncproxy(grasp_server)


def Plan(image, width_px=80):
    print(image.shape)
    image = image.astype('float32')
    # np.save('%.3f.npy' % width_px, image)
    return grasp_server.get_grasp(image, width_px)


class GrapsPanler(object):
    CROP_START = np.array([184, 0])
    CROP_SIZE = 480
    SCALE = 300 / CROP_SIZE

    def __init__(self, rgb, depth, camera, ur_status=None):
        self.rgb = rgb
        self.depth = depth
        self.camera = camera
        self.ur_status = ur_status
        self.plan_img = self.process(self.depth)
        self.width_px = self.get_width(self.plan_img, self.camera) * 300 / self.CROP_SIZE
        self.grasp_pyro = Plan(self.plan_img, self.width_px)

    def is_ready(self):
        return self.grasp_pyro.ready

    def get_grasp(self):
        if not self.is_ready():
            return None
        result = self.grasp_pyro.value
        print('result:', result)
        if not result[0]:
            return None
        p0, p1, d0, d1, q = result[1]
        p0 = self.process_to_original(p0)
        p1 = self.process_to_original(p1)
        self.grasp = Grasp2D.from_endpoint(p0, p1, d0, d1, q)
        return self.grasp

    @staticmethod
    def get_width(image, camera, width_m=0.14):
        d = image.mean()
        p0, _ = camera.project(np.array([-width_m/2, 0, d]))
        p1, _ = camera.project(np.array([width_m/2, 0, d]))
        return abs(p0[0] - p1[0])

    @classmethod
    def process(cls, image):
        img = image[cls.CROP_START[1]:cls.CROP_START[1] + cls.CROP_SIZE,
                    cls.CROP_START[0]:cls.CROP_START[0] + cls.CROP_SIZE]
        out_image = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        mask = ((out_image > 0.7) | (out_image < 0.1)).astype(np.uint8)
        # 消除掉夹爪在图像中的一点点影子
        mask[440:, :105] = 1
        mask[440:, 360:] = 1
        depth_scale = np.abs(out_image).max()
        out_image = out_image.astype(np.float32) / depth_scale
        out_image = cv2.inpaint(out_image, mask, 1, cv2.INPAINT_NS)
        out_image = out_image[1:-1, 1:-1]
        out_image = out_image * depth_scale
        # 要先降噪再缩放,缩放会带入噪声
        out_image = cv2.resize(out_image, (300, 300))
        return out_image

    def process_to_original(self, p):
        zoom = np.array(p) / self.SCALE
        crop = zoom + self.CROP_START
        return crop


class ServoGrasp(object):
    """ 伺服跟踪一个抓取,首先规划得到一个图像中的抓取位置,然后逆变换到世界坐标系,
    在世界坐标系中求出夹爪的姿态,沿夹爪z轴反推这个位姿作为新的相机坐标系位置,
    从而计算得到新的夹爪坐标系,执行,直到有新的规划结果,再伺服到新的位置，
    最后,当目标姿态和相机姿态一致且z轴距离小于一个指定值(0.35)时,
    规划一个夹爪的目标位姿,直接进行抓取,抓取后沿世界坐标z轴抬起
    需要实现的有
    1.抓取位姿映射器,实现抓取位姿在世界坐标、机械臂基坐标、机械手坐标和相机坐标之间的相互映射
    2.伺服追踪器,以恒定的速度移动工具到指定位置,且可在中途改变目标位置,并保证轨迹连续
    3.抓取规划器,输入图像和相机参数得到一个相机坐标系下的抓取位姿
    """

    def __init__(self, camera, urs, tf):
        self.camera = camera
        _ ,  _ , table_depth = self.camera.read(5)
        self.table_depth = table_depth
        # if table_depth:
        print("get it ",self.table_depth)
        self.urs = urs
        self.tf = tf
        self.gripper = Robotiq85()
        # self.gripper.reset()
        # self.gripper.activate(2)
        self.running = False
        self.grasp_planer = None
        self.grasp = None
        self.shot_one = False
        self.root = tk.Tk()  # 创建主窗体
        self.root.title('CameraImage')
        self.canvas = tk.Canvas()  # 创建一块显示图形的画布
        self.create_form()  # 将figure显示在tkinter窗体上面
        self.auto = tk.IntVar()
        self.add_button()
        self.video_loop()
        self.root.mainloop()

    def video_loop(self):
        success = False
        while not success:
            success, rgb, depth = self.camera.read(5)
        if self.running:
            if self.grasp_planer is None:
                ur_status = self.urs.ur_status()
                self.grasp_planer = GrapsPanler(rgb, depth, self.camera, ur_status)
            elif self.grasp_planer.is_ready():
                self.grasp = self.grasp_planer.get_grasp()
                if self.grasp is not None:
                    T_g_b = self.grasp_planer.ur_status
                    print(self.grasp.depth)
                    # 23.5,直接下去抓
                    if self.grasp.depth <= 0.29:
                        # T_g_w = self.tf.grasp2d_to_matrix(self.grasp, T_g_b, tcp='gripper')
                        # self.grasp_it(T_g_w)
                        self.running = False
                    else:
                        # 当前抓取深度向下10cm
                        h_offset = -max((self.grasp.depth - 0.3), 0.2)
                        print('h_offset', h_offset)
                        T_c_w, w = self.tf.grasp2d_to_matrix(
                            self.grasp, T_g_b, h=0, tcp='camera')
                        T_g_b = self.tf.grasp_from_camera(T_c_w)
                        self.urs.servo_to(T_g_b)
                self.grasp_planer = None
        if self.shot_one:
            im = GrapsPanler.process(depth)
            w = GrapsPanler.get_width(im, self.camera) * 300 / 480
            np.save('npy/width:%.3f.npy' % w, im)
            self.shot_one = False
        plt.clf()
        plt.axis('off')
        plt.imshow(rgb)
        self.plot_rect([184,0], 480, 480)
        plt.text(50, 50, 'ON' if self.running else 'OFF', size=150,
                 color='r', style="italic", weight="light")
        if self.grasp is not None:
            self.plot_grasp(self.grasp)
            plt.text(550, 50, 'q:%.4f' % self.grasp.quality, size=150,
                     color='r', style="italic", weight="light")
        self.canvas.draw()
        # 30ms后重复执行
        self.root.after(1, self.video_loop)
    
    def plot_rect(self, start, width, height):
        plt.plot((start[0], start[0]), (start[1],
                                        start[1]+height), 'r-', linewidth=8)
        plt.plot((start[0]+width, start[0]+width),
                 (start[1], start[1]+height), 'r-', linewidth=8)
        plt.plot((start[0], start[0]+width),
                 (start[1], start[1]), 'r-', linewidth=8)
        plt.plot((start[0], start[0]+width), (start[1] +
                                              height, start[1]+height), 'r-', linewidth=8)

    def grasp_it(self, T_g_w):
        T_g_b = self.tf.base_from_world(T_g_w)
        T_g_w1 = self.tf.matrix_translation(T_g_w, z=-0.07)
        T_g_b1 = self.tf.base_from_world(T_g_w1)
        self.urs.servo_to(T_g_b1)
        while self.urs.is_run():
            time.sleep(0.01)
        self.urs.servo_to(T_g_b)
        while self.urs.is_run():
            time.sleep(0.01)
        self.gripper.close()
        time.sleep(0.5)
        T_g_w1 = np.array([[0, 1, 0, 0.3],
                           [1, 0, 0, 0],
                           [0, 0, -1, 0.4],
                           [0, 0, 0, 1]])
        # T_g_w1 = T_g_w.copy()
        # T_g_w1[2, 3] = T_g_w1[2, 3] + 0.3
        T_g_b1 = self.tf.base_from_world(T_g_w1)
        # self.urs.set_speed(0.2)
        self.urs.movel_to(T_g_b1)
        # self.urs.set_speed(0.1)

    def plot_grasp(self, g, offset=[0, 0]):
        """ 使用plt在图像上展示一个夹爪 """
        def plot_2p(p0, p1, mode='r', width=None):
            p0 -= offset
            p1 -= offset
            x = [p0[0], p1[0]]
            y = [p0[1], p1[1]]
            plt.plot(x, y, mode, linewidth=width)

        def plot_center(center, axis, length, mode='r', width=2):
            axis = axis / np.linalg.norm(axis)
            p0 = center - axis * length / 2
            p1 = center + axis * length / 2
            plot_2p(p0, p1, mode, width)

        p0, p1, _, _ = g.endpoints
        axis = [g.axis[1], -g.axis[0]]
        plot_2p(p0, p1, 'r--', width=5)
        plot_center(p0, axis, g.width_px/5, width=8)
        plot_center(p1, axis, g.width_px/5, width=8)
        plt.plot(*(g.center - offset), 'bo')

    def start(self):
        self.running = True

    def stop(self):
        self.urs.clear()
        self.running = False
        self.grasp = None

    def create_form(self):
        f = plt.figure(num=2, figsize=(self.camera.width//10, self.camera.height//10), dpi=8)
        f.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # 把绘制的图形显示到tkinter窗口上
        self.canvas = FigureCanvasTkAgg(f, self.root)
        self.canvas.draw()  # 以前的版本使用show()方法，matplotlib 2.2之后不再推荐show（）用draw代替，但是用show不会报错，会显示警告
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def shot(self):
        self.shot_one = True

    def add_button(self):
        """ 增加三个按键 """
        button_frame = tk.Frame(self.root)
        button_frame.pack()
        b0 = tk.Button(button_frame, text='Start', font=('Arial', 12),
                       width=12, height=3, command=self.start)
        b0.pack(side=tk.LEFT)
        b1 = tk.Button(button_frame, text='Stop', font=('Arial', 12),
                       width=12, height=3, command=self.stop)
        b1.pack(side=tk.LEFT)
        b2 = tk.Button(button_frame, text='Open', font=('Arial', 12),
                       width=12, height=3, command=self.gripper.open)
        b2.pack(side=tk.LEFT)
        b4 = tk.Button(button_frame, text='Shot', font=('Arial', 12),
                       width=12, height=3, command=self.shot)
        b4.pack(side=tk.LEFT)
        b3 = tk.Button(button_frame, text='Exit', font=('Arial', 12),
                       width=12, height=3, command=self.root.quit)
        b3.pack(side=tk.LEFT)


if __name__ == "__main__":
    insces_path = os.path.join(root_path, 'cameras/images/realsense-d')
    # intr = None
    cam2base = np.load(os.path.join(root_path, 'cameras/arm_images/realsense-d/cam2base.npy'))
    camera = RealsenseSensor(align_to='depth', use='depth', insces_path=insces_path)
    camera.start()
    tf = TFMapper_EyeInHand(camera, cam2base)
    urs = URServo('192.168.1.101', speed=0.1)
    UI = ServoGrasp(camera, urs, tf)
    camera.stop()
    urs.close()
