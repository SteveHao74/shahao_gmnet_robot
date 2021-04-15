'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2019-11-04 13:14:06
@LastEditTime : 2020-01-04 00:00:33
@LastEditors  : Lai
'''
import os
import sys
import shutil
import cv2
import cv2.aruco as aruco
from glob import glob
import numpy as np
import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg   # NavigationToolbar2TkAgg
file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(file_path, '..'))
print(root_path)
sys.path.append(root_path)
try:
    from utils.primesense_sensor import PrimesenseSensor
except:
    print('import utils error')


# 标定物的尺寸
OBJ_SIZE = (7, 7)
# 标定点之间的距离, 单位mm
# 对相机的内参没有影响, 但是对目标相对相机位置有影响
OBJ_DIS = 30


class Calibrater(object):
    def __init__(self, out_path, camera, obj_size=(7, 7), obj_dis=30):
        self.save = False
        self.obj_size = obj_size
        self.obj_dis = obj_dis
        self.images = []
        self.auto_counter = 0
        self.out_path = os.path.abspath(out_path)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        self.root = tk.Tk()  # 创建主窗体
        self.root.title('Image')
        self.canvas = tk.Canvas()  # 创建一块显示图形的画布
        self.create_form()  # 将figure显示在tkinter窗体上面
        self.auto = tk.IntVar()
        self.add_button()
        self.camera = camera
        self.video_loop()
        self.root.mainloop()

    def video_loop(self):
        if self.auto.get():
            if len(self.images) >= 30:
                self.calibrate()
                self.auto.set(0)
            self.auto_counter = self.auto_counter + 1
            if self.auto_counter >= 15:
                self.auto_counter = 0
                self.save = True
        rgb = self.camera.read_color(10)
        bgr = rgb[..., ::-1].copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict,  parameters=parameters)
        if ids is not None:
            # rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.08, self.mtx, self.dist)
            # for i in range(rvec.shape[0]):
            # aruco.drawAxis(bgr, self.mtx, self.dist, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(bgr, corners, ids)
        if self.save:
            if ids is not None and len(ids) == 4:
                save_path = os.path.join(self.out_path, f'{len(self.images):03d}.jpg')
                cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                self.images.append(rgb.copy())
                print('save to:', save_path)
            else:
                print('Chessboard must in the photo')
            self.save = False
        plt.clf()
        plt.imshow(bgr[..., ::-1])
        plt.text(50, 50, f'num: {len(self.images):02d}', size=15,
                 color='r', style="italic", weight="light")
        plt.axis('off')
        self.canvas.draw()
        # 30ms后重复执行
        self.root.after(30, self.video_loop)

    def save_photo(self):
        """ 保存当前图像 """
        self.save = True

    def clean(self):
        self.images = []
        shutil.rmtree(self.out_path)
        os.makedirs(self.out_path)

    def calibrate(self):
        if len(self.images) < 10:
            print('need more image for calibrate!')
        else:
            mtx, dist, error = self.calibration(self.images, self.obj_size, self.obj_dis)
            np.save(os.path.join(self.out_path, 'mtx.npy'), mtx)
            np.save(os.path.join(self.out_path, 'dist.npy'), dist)

    def create_form(self):
        f = plt.figure(num=2, figsize=(8, 6), dpi=80)
        # 把绘制的图形显示到tkinter窗口上
        self.canvas = FigureCanvasTkAgg(f, self.root)
        self.canvas.draw()  # 以前的版本使用show()方法，matplotlib 2.2之后不再推荐show（）用draw代替，但是用show不会报错，会显示警告
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def add_button(self):
        """ 增加三个按键 """
        button_frame = tk.Frame(self.root)
        button_frame.pack()
        b0 = tk.Button(button_frame, text='Clean', font=('Arial', 12),
                       width=12, height=3, command=self.clean)
        b0.pack(side=tk.LEFT)
        b1 = tk.Button(button_frame, text='Save', font=('Arial', 12),
                       width=12, height=3, command=self.save_photo)
        b1.pack(side=tk.LEFT)
        b2 = tk.Button(button_frame, text='Calib', font=('Arial', 12),
                       width=12, height=3, command=self.calibrate)
        b2.pack(side=tk.LEFT)
        b3 = tk.Button(button_frame, text='Exit', font=('Arial', 12),
                       width=12, height=3, command=self.root.quit)
        b3.pack(side=tk.LEFT)
        b4 = tk.Checkbutton(button_frame, text='auto', font=('Arial', 12),
                            variable=self.auto, onvalue=True, offvalue=False)
        b4.pack(side=tk.LEFT)

    @staticmethod
    def calibration(images, obj_size=(7, 7), obj_dis=30):
        """ 进行相机标定
        images: RGB格式的一些图片数组
        """
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_50)
        parameters = aruco.DetectorParameters_create()
        board = aruco.GridBoard_create(2, 2, 0.08, 0.01, aruco_dict)
        charucoCorners = []
        charucoIds = []
        marknums = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                img, aruco_dict,  parameters=parameters)
            corners, ids, rejectedImgPoints, recoveryIdxs = aruco.refineDetectedMarkers(
                img, board, corners, ids, rejectedImgPoints)
            if ids is not None:
                charucoCorners.extend(corners)
                charucoIds.extend(ids)
                marknums.append(len(ids))
        charucoCorners = np.concatenate(charucoCorners)
        charucoIds = np.concatenate(charucoIds)
        marknums = np.array(marknums)
        ret, mtx, dist, rvecs, tvecs, _, _, perViewErrors = aruco.calibrateCameraArucoExtended(
            charucoCorners, charucoIds, marknums, board, img.shape[:2][::-1], None, None)
        print(ret)
        print("mtx:\n", mtx)      # 内参数矩阵
        print("dist:\n", dist)   # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
        # print("rvecs:\n", rvecs[0])   # 旋转向量  # 外参数
        # print("tvecs:\n", tvecs[0])  # 平移向量  # 外参数
        mean_error = np.mean(perViewErrors)
        print("total error: {:.4f}(pixel)".format(mean_error))
        return mtx, dist, mean_error


def calibration_in_file(file_path, obj_size=(7, 7), obj_dis=30):
    files = glob(os.path.join(file_path, '*.jpg'))
    images = []
    for f in files:
        img = cv2.imread(f)
        cv2.imshow('img', img)
        cv2.waitKey(100)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    Calibrater.calibration(images, obj_size, obj_dis)


if __name__ == "__main__":
    out_path = os.path.join(root_path, 'images')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with PrimesenseSensor() as camera:
        UI = Calibrater(out_path, camera, OBJ_SIZE, OBJ_DIS)
