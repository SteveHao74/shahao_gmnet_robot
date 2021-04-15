'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2019-11-04 13:14:06
@LastEditTime : 2020-01-15 20:53:11
@LastEditors  : Lai
'''
import os
import sys
import shutil
import cv2
import cv2.aruco as aruco
import time
import numpy as np
import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg   # NavigationToolbar2TkAgg
from armeye_calibration import eye_arm_calibrate
file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(file_path, '..'))
print(root_path)
sys.path.append(root_path)
try:
    from utils.ur_servo import URServo
    from utils.realsenseL515 import RealsenseSensor
except Exception as e:
    print('import utils error')
    raise e


# 标定物的尺寸
OBJ_SIZE = (7, 7)
# 标定点之间的距离, 单位mm
# 对相机的内参没有影响, 但是对目标相对相机位置有影响
OBJ_DIS = 30


class SampleDataUI(object):
    def __init__(self, out_path, camera, tcps_path):
        self.save = False
        self.image_num = 0
        self.auto_counter = 0
        self.out_path = os.path.abspath(out_path)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        self.urs = URServo('192.168.3.100')
        self.is_move = False
        self.wait = 0
        self.mtx = camera.mtx
        self.dist = camera.dist
        print('mtx:\n', self.mtx)
        print('dist:\n', self.dist)
        self.camera = camera
        self.tcps_path = tcps_path
        self.make_ui()
        self.update_ur_ip()
        self.video_loop()
        self.root.mainloop()

    def make_ui(self):
        self.root = tk.Tk()  # 创建主窗体
        self.root.title('Calibrater')
        self.auto = tk.IntVar()
        f = tk.Frame(self.root)
        f.pack(side=tk.LEFT)
        self.create_form(f).pack(side=tk.TOP, ipadx=5, ipady=5,
                                 padx=5, pady=5)  # 将figure显示在tkinter窗体上面
        self.add_button(f).pack(pady=5)
        self.add_ur_frame(self.root).pack(side=tk.RIGHT, ipadx=5)
        self.root.protocol("WM_DELETE_WINDOW", self.exit_)

    def add_ur_frame(self, frame):
        def add_info(frame, name, row):
            l0 = tk.Label(frame, text=f'{name}:', font=('Arial', 12))
            l0.grid(row=row, sticky=tk.E, ipady=5)
            text = tk.StringVar()
            text.set('%03.3f' % (0))
            l1 = tk.Label(frame, textvariable=text, font=('Arial', 12))
            l1.grid(row=row, column=1, sticky=tk.W)
            return text
        ur_frame = tk.Frame(frame)
        l0 = tk.Label(ur_frame, text='UR5 Info', font=('Arial', 16))
        l0.grid(row=0, ipady=5, columnspan=2)
        l1 = tk.Label(ur_frame, text='IP:', font=('Arial', 12))
        l1.grid(row=1, sticky=tk.E, ipady=5)
        self.ur_IP = tk.StringVar(value='192.168.3.100')
        e1 = tk.Entry(ur_frame, bd=1, textvariable=self.ur_IP, font=('Arial', 12), width=15)
        e1.grid(row=1, column=1)
        l2 = tk.Label(ur_frame, text='speed:', font=('Arial', 12))
        l2.grid(row=2, sticky=tk.E, ipady=5)
        self.speed = tk.StringVar(value='0.1')
        e2 = tk.Entry(ur_frame, bd=1, textvariable=self.speed, font=('Arial', 12), width=10)
        e2.grid(row=2, column=1)
        self.ur_text = dict()
        for i, n in enumerate('X Y Z rx ry rz'.split()):
            text = add_info(ur_frame, n, i+3)
            self.ur_text[n] = text
        return ur_frame

    def create_form(self, frame):
        f = plt.figure(num=2, figsize=(self.camera.width//10, self.camera.height//10), dpi=10)
        f.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # 把绘制的图形显示到tkinter窗口上
        self.canvas = FigureCanvasTkAgg(f, frame)
        self.canvas.draw()  # 以前的版本使用show()方法，matplotlib 2.2之后不再推荐show（）用draw代替，但是用show不会报错，会显示警告
        return self.canvas.get_tk_widget()

    def add_button(self, frame):
        """ 增加三个按键 """
        button_frame = tk.Frame(frame)
        b0 = tk.Button(button_frame, text='Clean', font=('Arial', 12),
                       width=12, height=3, command=self.clean)
        b0.pack(side=tk.LEFT, padx=5)
        b1 = tk.Button(button_frame, text='Save', font=('Arial', 12),
                       width=12, height=3, command=self.save_photo)
        b1.pack(side=tk.LEFT, padx=5)
        b2 = tk.Button(button_frame, text='Next', font=('Arial', 12),
                       width=12, height=3, command=self.next_pose)
        b2.pack(side=tk.LEFT, padx=5)
        b3 = tk.Button(button_frame, text='Exit', font=('Arial', 12),
                       width=12, height=3, command=self.exit_)
        b3.pack(side=tk.LEFT, padx=5)
        b4 = tk.Checkbutton(button_frame, text='auto', font=('Arial', 12),
                            variable=self.auto, onvalue=True, offvalue=False)
        b4.pack(side=tk.LEFT, padx=5)
        return button_frame

    def next_pose(self):
        """ 机械臂运动到下一个位置并拍照 """
        if not self.save:
            try:
                tcp = np.load(os.path.join(self.tcps_path, f'{self.image_num:03d}_tcp_pose.npy'))
            except:
                print('tcp文件读取失败')
                return False
            self.movel_to(tcp)
            self.save = True
            return True

    def movel_to(self, tcp):
        """ 移动到下一个位姿 """
        self.urs.movel_to(tcp, min(float(self.speed.get().strip()), 0.2))
        self.is_move = True
        self.target = tcp
        self.wait = 3

    def update_ur_ip(self):
        ip = self.ur_IP.get().strip()
        if ip != self.urs.host:
            self.urs.close()
            self.urs = URServo(ip)

    def upadte_ur_status(self):
        try:
            # self.rq_width.set('%03.3f' % self.gripper.get_pos())
            ur_s = self.urs.ur_status(matrix=False)
            for i, n in enumerate('X Y Z rx ry rz'.split()):
                self.ur_text[n].set('%03.3f' % ur_s[i])
        except:
            # print('arm or gripper read error', self.ur_IP.get().strip())
            return False, None
        else:
            return True, ur_s

    def video_loop(self):
        self.update_ur_ip()
        if self.auto.get():
            if not self.save and not self.is_move:
                if not self.next_pose():
                    self.auto.set(0)
                    self.movel_to(np.load(os.path.join(self.tcps_path, '000_tcp_pose.npy')))
        if self.is_move:
            if np.linalg.norm(self.urs.ur_status(False) - self.target) < 1e-3:
                print('wait:', self.wait)
                self.wait -= 1
            if self.wait == 0:
                self.is_move = False
        arm_r, arm_s = self.upadte_ur_status()
        rgb, depth = self.camera.frame(10)
        depth = depth * 1000
        bgr = rgb[..., ::-1].copy()
        if not self.is_move and self.save:
            save_path = os.path.join(self.out_path, f'{self.image_num:03d}.png')
            arm_path = os.path.join(self.out_path, f'{self.image_num:03d}_tcp_pose.npy')
            depth_path = os.path.join(self.out_path, f'{self.image_num:03d}_depth.npy')
            cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            np.save(arm_path, arm_s)
            np.save(depth_path, depth)
            print('save to:', save_path)
            self.save = False
            self.image_num += 1
        plt.clf()
        plt.axis('off')
        plt.imshow(bgr[..., ::-1])
        plt.text(50, 50, f'num: {self.image_num:02d}', size=150,
                 color='r', style="italic", weight="light")
        self.canvas.draw()
        # 30ms后重复执行
        self.root.after(10, self.video_loop)

    def save_photo(self):
        """ 保存当前图像 """
        self.save = True

    def clean(self):
        self.image_num = 0
        shutil.rmtree(self.out_path)
        os.makedirs(self.out_path)

    def exit_(self):
        self.urs.close()
        time.sleep(1)
        self.root.quit()


if __name__ == "__main__":
    out_path = os.path.join(root_path, 'cameras/sample_data')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    tcps_path = os.path.join(root_path, 'cameras/tcp_poses')
    with RealsenseSensor(align_to='color', use='color') as camera:
        UI = SampleDataUI(out_path, camera, tcps_path)
