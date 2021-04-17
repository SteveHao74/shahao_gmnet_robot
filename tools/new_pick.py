'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2020-01-16 20:03:33
@LastEditTime : 2020-01-17 14:30:14
@LastEditors  : Lai
'''
import os
import sys
from cv2 import  cv2
import time
import Pyro4
import numpy as np
import tkinter as tk
from matplotlib import pyplot as plt
from servo_grasp import ServoGrasp
file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(file_path, '..'))
print(root_path)
sys.path.append(root_path)
try:
    from utils.realsenseL515 import RealsenseSensor
    from utils.grasp_2d import Grasp2D
    from utils.tf_mapper import TFMapper_EyeToHand
    from utils.ur_servo import URServo
except Exception as e:
    print('import utils error')
    raise e

Pyro4.config.SERIALIZER = 'pickle'
grasp_server = Pyro4.Proxy("PYRO:grasp@10.12.120.55:6665")
Pyro4.asyncproxy(grasp_server)

total_length = []
height_min = 0.146#一定要高于这个高度，避免爪子碰到

HHHH = 40
def Plan(image):
    print(image.shape)
    image = image.astype('float32')
    np.save('ddd.npy', image)
    return grasp_server.plan(image, 70)

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

class GrapsPanler(object):
    CROP_START = np.array([400, 200])#200, 50
    CROP_SIZE = 480
    SCALE = 300 / CROP_SIZE

    def __init__(self, rgb, depth, camera, ur_status=None):
        self.rgb = rgb
        self.depth = depth
        self.camera = camera
        self.ur_status = ur_status
        self.image = self.process(depth)
        self.mean = np.mean(self.image)
        self.std = np.std(self.image)        
        # self.min_depth = np.min(b)#np.min(image[np.nonzero(image)])
        print('----------------------------')
        print(self.image)
        print('图像尺寸:', self.image.shape)
        print('----------------------------')
        print('深度均值:', self.mean)
        print('----------------------------')
        print('深度标准差:', self.std)
        print('----------------------------')
        self.grasp_pyro = Plan(self.image)
        self.min_depth    = self.get_min_depth()

    def get_min_depth(self):
        img_GaussianBlur = cv2.GaussianBlur(self.image,(5,5),0)
        img_medianBlur = cv2.medianBlur(img_GaussianBlur,5)#,(5,5))
        a = img_medianBlur[np.nonzero(img_medianBlur)]
        b = np.array([n for n in a if n>0.5])
        min_depth = np.min(b)
        print('最小深度:', min_depth)
        return min_depth

    def is_ready(self):
        return self.grasp_pyro.ready

    def get_grasp(self):
        if not self.is_ready():
            return None
        print('---------------------------')
        print('服务器返回')
        print('---------------------------')
        try:
            result = self.grasp_pyro.value
        except Exception as e:
            for _ in range(20):
                print('\033[31;41m 规划失败!!!!!! \033[0m', e)
            return None
        print('服务器返回的结果为：', result)
        if result is None:
            return None
        p0, p1, d0, d1, q = result
        p0 = self.process_to_original(p0)#这里是要把在300x300的小图下的抓取点坐标还原到整个摄像头的原始视野中
        p1 = self.process_to_original(p1)
        self.grasp = Grasp2D.from_endpoint(p0, p1, d0, d1, q)
        return self.grasp

    @classmethod
    def process(cls, image):
        img = image[cls.CROP_START[1]:cls.CROP_START[1] + cls.CROP_SIZE,
                    cls.CROP_START[0]:cls.CROP_START[0] + cls.CROP_SIZE]
        img = cv2.resize(img, (300, 300))
        out_image = cv2.copyMakeBorder(
            img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
        # mask = ((out_image > 1.2) | (out_image < 0.1)).astype(np.uint8)
        mask = (out_image < 0.1).astype(np.uint8)
        depth_scale = np.abs(out_image).max()
        out_image = out_image.astype(np.float32) / depth_scale
        out_image = cv2.inpaint(out_image, mask, 1, cv2.INPAINT_NS)
        out_image = out_image[10:-10, 10:-10]
        out_image = out_image * depth_scale
        return out_image

    @classmethod
    def process_rgb(cls, image):
        img = image[cls.CROP_START[1]:cls.CROP_START[1] + cls.CROP_SIZE,
                    cls.CROP_START[0]:cls.CROP_START[0] + cls.CROP_SIZE]
        img = cv2.resize(img, (300, 300))
        return img

    def process_to_original(self, p):#这里是要把在300x300的小图下的抓取点坐标还原到整个摄像头的原始视野中
        zoom = np.array(p) / self.SCALE
        crop = zoom + self.CROP_START
        return crop


class NewPick(ServoGrasp):
    # put_pose = np.array([[0, 1, 0, 0.4],
    #                      [1, 0, 0, -0.2],
    #                      [0, 0, -1, 0.4],
    #                      [0, 0, 0, 1]])

    def __init__(self, camera, urs, tf):
        self.planing = False
        self.grasp = None
        self.target = None
        self.ex = False
        self.shot_one = False
        self.image_num = 0
        self.init_pose = np.array([0.26964,-0.10738,0.39318,2.9278,1.0482,0.0885])#根据示教板的读数确定的初始位姿
        self.init_pose_matrix = vector_to_matrix(self.init_pose) #使用opencv库将旋转向量表示法转换为旋转矩阵表示法
        print(self.init_pose_matrix )
        urs.movel_to(self.init_pose, v=0.25)
        super().__init__(camera, urs, tf)


    def video_loop(self):
        success = False
        while not success:
            success, rgb, depth = self.camera.read(5)
            # print("rgb is",rgb)   
            # print(rgb.shape)
        if self.planing:
            if self.grasp_planer is None:
                # print("get it ",self.table_depth)
                ur_status = self.urs.ur_status()
                depths = [depth] + [self.camera.read_depth(1) for _ in range(4)]
                # depths =  [self.camera.read_depth(1) for _ in range(4)]
                depth = np.mean(depths, axis=0)#计算几张深度图矩阵的均值，depth存着的是均值
                mean = np.mean(depth)
                new_std = np.std(depth)
                # print("平均值",mean)
                # print("标准差",new_std)
            
                # fig = plt.figure()
                # ax1 = fig.add_subplot(221)
                # ax1.imshow(depth, cmap= plt.cm.gray)
                # depth =  (depth-self.table_depth)#/new_std* 0.005
                # ax2 = fig.add_subplot(222)
                # ax2.imshow(self.table_depth, cmap= plt.cm.gray)
                # ax3 = fig.add_subplot(223)
                # ax3.imshow(depth, cmap= plt.cm.gray)
                # plt.show()

                # print("-------------shahaoshahao",depth)
                self.grasp_planer = GrapsPanler(
                    rgb, depth, self.camera, ur_status)
                self.mean= self.grasp_planer.mean
                self.std= self.grasp_planer.std
                self.grasp_depth = self.grasp_planer.min_depth
                self.processed_depth_img = self.grasp_planer.image#得到经过中值滤波的深度图
                

            elif self.grasp_planer.is_ready():
                ggg = self.grasp_planer.get_grasp()
                if ggg is None:
                    print('\033[31;41m 规划失败!!!!!! \033[0m')
                self.grasp = ggg
                self.grasp_planer = None
                self.planing = False
                depth_point = self.grasp.center - GrapsPanler.CROP_START
                print("抓取点坐标",depth_point)
                self.grasp_depth=self.processed_depth_img[int(depth_point[0]/1.6),int(depth_point[1]/1.6)] 
                print("抓取的宽度为",self.grasp_depth)

        if self.ex:
            # 如果没有目标或者达到目标则切换下一个目标
            if self.target is None or self.is_get_target():
                if self.targets:
                    self.target = self.targets.pop(0)
                    # a=np.array([0.29964,-0.10738,0.39318,2.9278,1.0482,0.0885])
                    # urs.movel_to(a, v=0.1)
                    self.start_target()
                else:
                    self.target = None
        # save_path="/home/shahao/gmnet_robot_shahao"            
        if self.shot_one:
            np.save('npy/%03d_raw.npy'%self.image_num, depth)
            np.save('npy/%03d.npy'%self.image_num, GrapsPanler.process(depth))
            cv2.imwrite('npy/%03d.png'%self.image_num, GrapsPanler.process_rgb(rgb))
            self.image_num += 1
            self.shot_one = False

        plt.clf()
        plt.axis('off')
        plt.imshow(rgb)
        # plt.imshow(GrapsPanler.process(depth))
        self.plot_rect(GrapsPanler.CROP_START, GrapsPanler.CROP_SIZE, GrapsPanler.CROP_SIZE)
        # plt.colorbar()
        plt.text(50, 50, 'ON' if self.planing else 'OFF', size=150,
                 color='r', style="italic", weight="light")
        # plt.text(350, 50, 'w:%.4f' % self.get_width(depth), size=150,
        #          color='r', style="italic", weight="light")
        if self.grasp is not None:
            self.plot_grasp(self.grasp)
            plt.text(50, 420, 'current_len:%.4f' % self.grasp.quality, size=160,
                     color='r', style="italic", weight="light")
            total_length.append(self.grasp.quality)
            plt.text(50, 440, 'min_len:%.4f' % max(total_length), size=160,
                     color='g', style="italic", weight="light")
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

    def plan(self):
        self.planing = True

    def start_target(self):
        if self.target[0] == 'move':
            self.urs.movel_to(self.target[1], v=0.1)
        elif self.target[0] == 'close':
            self.gripper.goto(self.target[1])
            if self.target[1] == 0:
                time.sleep(0.5)
        elif self.target[0] == 'open':
            self.gripper.open()
            time.sleep(1)

    def is_get_target(self):
        if self.target is None:
            return True
        elif self.target[0] == 'move':
            ur_status = self.urs.ur_status()
            if np.sum(np.abs(self.target[1] - ur_status)) < 0.001:
                return True
            else:
                return False
        elif self.target[0] == 'close' or self.target[0] == 'open':
            return True

    def execute(self):
        if self.grasp is not None:
            self.ex = True
            ur_status = self.urs.ur_status()
            print('111111111111111,ur_status',ur_status)

            g_world, width = self.tf.grasp2d_to_matrix(self.grasp, ur_status,grasp_depth = self.grasp_depth)#得到了目标抓取在世界坐标系下的位姿，和抓取宽度（两个抓取点间的距离）
            #gp_world = self.tf.matrix_translation(g_world, z=height_min)
            print("g_world",g_world)
            g_base = self.tf.base_from_world(g_world)
            g_base[2,3] =g_base[2,3]+0.16# 0.157#
            if g_base[2,3] < 0.157:
                g_base[2,3] = 0.157
            gtop_base =  g_base.copy()
            gtop_base[2,3] =  self.init_pose[2] 
            targets = []
            print("top",gtop_base)
            print("zhuaquweizi",g_base)
            print("width",width)
            targets.append(['move',gtop_base])
            # targets.append(['close', 0.05])
            targets.append(['open', None])
            targets.append(['move', g_base])
            targets.append(['close', 0])
            # targets.append(['close', 0.04])
            targets.append(['move',gtop_base])
            targets.append(['move', self.tf.base_from_world(self.init_pose_matrix)])
            targets.append(['open', None])
            self.targets = targets
    
    def shot(self):
        self.shot_one = True

    def add_button(self):
        """ 增加三个按键 """
        button_frame = tk.Frame(self.root)
        button_frame.pack()
        b0 = tk.Button(button_frame, text='Plan', font=('Arial', 12),
                       width=12, height=3, command=self.plan)
        b0.pack(side=tk.LEFT)
        b1 = tk.Button(button_frame, text='Execute', font=('Arial', 12),
                       width=12, height=3, command=self.execute)
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
    # insces_path = os.path.join(root_path, 'cameras/images/primesense')
    insces_path = os.path.join(root_path, 'cameras/images/realsense')
    out_path = os.path.join(root_path, 'cameras/arm_images/realsense')
    # intr = None
    cam2base = np.load(os.path.join(
        root_path, 'cameras/arm_images/realsense/cam2base.npy'))
    camera = RealsenseSensor(align_to='color',use='color', insces_path=insces_path)
    # camera = PrimesenseSensor(auto_white_balance=True, insces_path=insces_path)
    camera.start()
    tf = TFMapper_EyeToHand(camera, cam2base)
    urs = URServo('192.168.1.101', speed=1)
    UI = NewPick(camera, urs, tf)
    camera.stop()
    urs.close()
