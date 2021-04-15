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
    from utils.ur_status import URStatus
    from utils.realsenseL515 import RealsenseSensor
except Exception as e:
    print('import utils error')
    raise e


# 标定物的尺寸
OBJ_SIZE = (7, 7)
# 标定点之间的距离, 单位mm
# 对相机的内参没有影响, 但是对目标相对相机位置有影响
OBJ_DIS = 30


class ArmEyeUI(object):
    def __init__(self, out_path, camera, mtx_path=None):
        self.save = False
        self.images = []
        self.arm_s = []
        self.auto_counter = 0
        try:
            self.ur_status = URStatus()
        except:
            print('check arm and gripper has connected !!!!!!')
        self.out_path = os.path.abspath(out_path)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        if mtx_path:
            self.mtx = np.load(os.path.join(mtx_path, 'mtx.npy'))
            self.dist = np.load(os.path.join(mtx_path, 'dist.npy'))
        else:
            self.mtx = camera.mtx
            self.dist = camera.dist
        print('mtx:\n', self.mtx)
        print('dist:\n', self.dist)
        self.camera = camera
        self.make_ui()
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
        e1 = tk.Entry(ur_frame, bd=1, textvariable=self.ur_IP, font=('Arial', 12), width=12)
        e1.grid(row=1, column=1)
        self.ur_text = dict()
        for i, n in enumerate('X Y Z rx ry rz'.split()):
            text = add_info(ur_frame, n, i+2)
            self.ur_text[n] = text
        l3 = tk.Label(ur_frame, text='Robotiq85', font=('Arial', 16))
        l3.grid(row=8, ipady=5, columnspan=2)
        self.rq_width = add_info(ur_frame, 'W', 9)
        b0 = tk.Button(ur_frame, text='open', font=('Arial', 12),
                       width=12, height=1, command=self.gripper_open)
        b0.grid(row=10, ipady=5, pady=5, columnspan=2)
        b1 = tk.Button(ur_frame, text='close', font=('Arial', 12),
                       width=12, height=1, command=self.gripper_close)
        b1.grid(row=11, ipady=5, pady=5, columnspan=2)
        l4 = tk.Label(ur_frame, text='Eye_Hand', font=('Arial', 16))
        l4.grid(row=12, ipady=5, columnspan=2)
        self.is_eye_in_hand = tk.IntVar()
        rb0 = tk.Radiobutton(ur_frame, text='eye_in_hand', value=1,
                             variable=self.is_eye_in_hand, font=('Arial', 12))
        rb0.grid(row=13, pady=5, columnspan=2)
        rb1 = tk.Radiobutton(ur_frame, text='eye_on_hand', value=0,
                             variable=self.is_eye_in_hand, font=('Arial', 12))
        rb1.grid(row=14, pady=5, columnspan=2)
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
        b2 = tk.Button(button_frame, text='Calib', font=('Arial', 12),
                       width=12, height=3, command=self.save_photo)
        b2.pack(side=tk.LEFT, padx=5)
        b3 = tk.Button(button_frame, text='Exit', font=('Arial', 12),
                       width=12, height=3, command=self.root.quit)
        b3.pack(side=tk.LEFT, padx=5)
        b4 = tk.Checkbutton(button_frame, text='auto', font=('Arial', 12),
                            variable=self.auto, onvalue=True, offvalue=False)
        b4.pack(side=tk.LEFT, padx=5)
        return button_frame

    def gripper_open(self):
        return 
        self.gripper.open()

    def gripper_close(self):
        return
        self.gripper.close()

    def upadte_ur_status(self):
        try:
            # self.rq_width.set('%03.3f' % self.gripper.get_pos())
            ur_s = self.ur_status.read('Tool_vector_actual', host=self.ur_IP.get().strip())
            for i, n in enumerate('X Y Z rx ry rz'.split()):
                self.ur_text[n].set('%03.3f' % ur_s[i])
        except:
            # print('arm or gripper read error', self.ur_IP.get().strip())
            return False, None
        else:
            return True, ur_s

    def video_loop(self):
        if self.auto.get():
            if len(self.images) >= 30:
                # self.calibrate()
                self.auto.set(0)
            self.auto_counter = self.auto_counter + 1
            if self.auto_counter >= 8:
                self.auto_counter = 0
                self.save = True
        arm_r, arm_s = self.upadte_ur_status()
        rgb, depth = self.camera.frame(10)
        bgr = rgb[..., ::-1].copy()
        # bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.save:
            save_path = os.path.join(self.out_path, f'{len(self.images):03d}.jpg')
            arm_path = os.path.join(self.out_path, f'{len(self.images):03d}_tcp_pose.npy')
            depth_path = os.path.join(self.out_path, f'{len(self.images):03d}_depth.npy')
            cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            np.save(arm_path, arm_s)
            np.save(depth_path, depth)
            self.images.append(rgb.copy())
            self.arm_s.append(np.array(arm_s))
            print('save to:', save_path)
            self.save = False
        plt.clf()
        plt.axis('off')
        plt.imshow(bgr[..., ::-1])
        plt.text(50, 50, f'num: {len(self.images):02d}', size=150,
                 color='r', style="italic", weight="light")
        self.canvas.draw()
        # 30ms后重复执行
        self.root.after(10, self.video_loop)

    def save_photo(self):
        """ 保存当前图像 """
        self.save = True

    def clean(self):
        self.images = []
        self.arm_s = []
        shutil.rmtree(self.out_path)
        os.makedirs(self.out_path)


if __name__ == "__main__":
    out_path = os.path.join(root_path, 'cameras/sample_data')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with RealsenseSensor(align_to='color',use='color') as camera:
        UI = ArmEyeUI(out_path, camera)

