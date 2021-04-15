'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2019-12-19 23:09:45
@LastEditTime : 2019-12-29 20:02:37
@LastEditors  : Lai
'''
import os
import sys
import time
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(file_path, '..'))
print(root_path)
sys.path.append(root_path)
gmnet_path = os.path.abspath(os.path.join(root_path, '../gmnet/tools'))
sys.path.append(gmnet_path)
try:
    from utils.ur_control import URSControl
    from utils.ur_status import URStatus
except:
    print('import utils error')

HOST = '192.168.1.102'
File = os.path.abspath(os.path.join(root_path, 'data_real1.csv'))
# V = 0.01
V = 0.2
P = 0.02
MULT_jointstate = 10000.0
MULT_time = 1000000.0

# x1 = cos(c1)
# y1 = sin(c1)
# x2 = cos(c2)
# y2 = sin(c2)
# xm = (x1+x2)/2
# ym = (y1+y2)/2
# cm = atan2(ym, xm);
# huanshou1 = [[102.26, -114.68, 84.78, -92.82, -94.06, 0],
#              [-20.6, -109.72, 84.27, -68.49, -80.80, 0],
#              [-10, -148, 82.07, -27, -80, 0],
#              [-10, -107, -48, -270, -80, 0]]
huanshou_wok_no_obstacle = [[-170, -81, 71, -80, -89, 0],
                            [-170, -72, -40, -170, 78, 0],
                            [-40, -72, -40, -170, 78, 0],
                            [33, -80, -89, -96, 87, 0]]
# 文件1的第一个角度[-118, -84, 94, -82, -130, 0]
huanshou_s3_1 = [[-166, -78, 57, 84, 91, 0],
                 [-118, -84, 94, -82, -130, 0]]
huanshou_s1_2 = [[-170, -81, 71, -80, -89, 0],
                 [-170, -72, -40, -170, 78, 0],
                 [-40, -72, -40, -170, 78, 0],
                 [33, -80, -89, -96, 87, 0]]
huanshou_s1 = [-118, -84, 94, -82, -130, 0]
huanshou_s3 = [-115, -82, 113, 30, 145, 0]


def load_traj(file):
    f = pd.read_csv(file, header=None)
    return np.array(f)


def traj_inter(traj, speed, proid=0.04):
    time_point = [0]
    t = 0
    for i in range(len(traj)-1):
        # t = t + np.linalg.norm(traj[i+1, -3:] - traj[i, -3:]) / speed
        t = t + np.linalg.norm(traj[i+1, :5] - traj[i, :5]) / speed
        time_point.append(t)
    f_inter = []
    for i in range(5):
        f = interp1d(time_point, traj[:, i], kind='cubic')
        f_inter.append(f)
    ps = (time_point[-1] - time_point[0]) // proid + 1
    t_out = np.linspace(time_point[0], time_point[-1], ps)
    out = np.zeros((len(t_out), ))
    for i in range(5):
        out = np.c_[f_inter[4-i](t_out), out]
    # return out
    return traj, time_point


def exec_file(file, urc, urs):
    traj = load_traj(file)
    tra = traj_inter(traj, V, P)
    urc.movej(tra[0], True, v=0.1)
    while True:
        q = urs.q_actual
        if np.linalg.norm(tra[0] - q) < 0.05:
            break
        time.sleep(0.1)
    print('start exec the traj')
    for trt in tra[1:]:
        start = time.time()
        urc.servoj(trt, t=P*2, lookahead=0.05, gain=200)
        while time.time() - start < P:
            pass


def movej(pose, urc, urs):
    urc.movej(pose, True, v=0.2)
    while True:
        q = urs.q_actual
        if np.linalg.norm(pose - q) < 0.05:
            break
        time.sleep(0.1)


def wok_no_obstacle():
    urc = URSControl(host=HOST)
    urs = URStatus(host=HOST)
    file1 = os.path.abspath(os.path.join(root_path, 'wok_no_obstacle1.csv'))
    file2 = os.path.abspath(os.path.join(root_path, 'wok_no_obstacle2.csv'))
    exec_file(file1, urc, urs)
    time.sleep(0.5)
    for p in huanshou_wok_no_obstacle:
        movej(np.deg2rad(p), urc, urs)
    exec_file(file2, urc, urs)
    time.sleep(0.5)
    for p in huanshou_wok_no_obstacle[::-1]:
        movej(np.deg2rad(p), urc, urs)


def wok_with_obstacle():
    urc = URSControl(host=HOST)
    urs = URStatus(host=HOST)
    file1 = os.path.abspath(os.path.join(root_path, 'shuaguo-data/wok_with_obstacle1.csv'))
    file2 = os.path.abspath(os.path.join(root_path, 'shuaguo-data/wok_with_obstacle2.csv'))
    file3 = os.path.abspath(os.path.join(root_path, 'shuaguo-data/wok_with_obstacle3.csv'))
    # movej([0.658906055, -2.327907738, -0.667480238, -2.23554566, 1.49226409, 0], urc, urs)
    # movej([-2.961381433, -1.484077897,
    #        1.592799243, -1.679517673, -1.570796327, 0], urc, urs)

    exec_file(file3, urc, urs)
    time.sleep(0.5)
    for p in huanshou_s3_1:
        movej(np.deg2rad(p), urc, urs)
    exec_file(file1, urc, urs)
    time.sleep(0.5)
    for p in huanshou_s1_2:
        movej(np.deg2rad(p), urc, urs)
    exec_file(file2, urc, urs)
    time.sleep(0.5)
    for p in huanshou_s1_2[::-1]:
        movej(np.deg2rad(p), urc, urs)
    movej(np.deg2rad(huanshou_s3), urc, urs)


def out():
    files = []
    files.append(os.path.abspath(os.path.join(root_path, 'shuaguo-data/wok_with_obstacle1.csv')))
    files.append(os.path.abspath(os.path.join(root_path, 'shuaguo-data/wok_with_obstacle2.csv')))
    files.append(os.path.abspath(os.path.join(root_path, 'shuaguo-data/wok_with_obstacle3.csv')))
    files.append(os.path.abspath(os.path.join(root_path, 'shuaguo-data/wok_no_obstacle1.csv')))
    files.append(os.path.abspath(os.path.join(root_path, 'shuaguo-data/wok_no_obstacle2.csv')))
    for f in files:
        traj = load_traj(f)
        tra, t = traj_inter(traj, V, P)
        tra = np.c_[tra, t]
        np.savetxt('out_' + os.path.basename(f), tra, delimiter=',', fmt='%.10f')


if __name__ == "__main__":
    # wok_with_obstacle()
    out()
