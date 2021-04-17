'''
@Description: 机械臂伺服控制
@Author: Lai
@Date: 2020-01-04 20:23:31
@LastEditTime : 2020-01-16 21:02:06
@LastEditors  : Lai
'''
import time
import numpy as np
from multiprocessing import Process, Pipe
from threading import Thread
from .ur_control import URControl
from .ur_status import URStatus
from .kinematics2 import InverseKinematicsUR5
from .kinematics1 import inv_kin, ur2np
from .tf_mapper import np2ur


class URServoProcess(Process):
    def __init__(self, pipe, host, ctrl_period=0.02, speed=0.5):
        super().__init__()
        self.pipe = pipe
        self.host = host
        self.urc = None
        self.urs = None
        self.traj = []
        self.stop = True
        self.ctrl_period = ctrl_period
        self.speed = speed
        self.ur5_ik = InverseKinematicsUR5()
        self.ur5_ik.setJointWeights([6, 5, 4, 3, 2, 1])
        self.ur5_ik.setJointLimits(-2*np.pi, 2*np.pi)

    def run(self):
        self.stop = False
        self.urc = URControl(self.host)
        self.urs = URStatus(self.host)
        thr = Thread(target=self.servo_timer)
        thr.start()
        while True:
            # 这里通信通过两个参数,状态码和数据
            s, m = self.pipe.recv()
            if s == 'stop':
                break
            elif s == 'goto':
                q_now = self.urs.q_actual
                ur_q = self.ik(m, q_now)
                if len(self.traj) >= 2:
                    d_now = np.array(self.traj[1]) - np.array(self.traj[0])
                    d_now = d_now / self.ctrl_period
                else:
                    d_now = np.zeros((6,))
                print('from {} to {}'.format(q_now, ur_q))
                self.traj = self.get_traj(q_now, ur_q, d_now, self.speed, self.ctrl_period)
            elif s == 'urs':
                self.pipe.send(('urs', self.urs.Tool_vector_actual))
            elif s == 'urq':
                self.pipe.send(('urs', self.urs.q_actual))
            elif s == 'is_run':
                self.pipe.send(('is_run', len(self.traj)))
            elif s == 'set_speed':
                self.speed = float(m)
            elif s == 'clear':
                self.traj = []
            elif s == 'movel':
                self.traj = []
                self.urc.movel(m[0], v=m[1])
        self.stop = True
        if thr.is_alive():
            thr.join()

    def servo_timer(self):
        period = self.ctrl_period*4
        while True:
            t = time.time()
            if self.stop:
                break
            if self.traj:
                tra = self.traj.pop(0)
                # print(tra)
                self.urc.servoj(tra, period)
            while time.time() - t < self.ctrl_period:
                pass

    @staticmethod
    def get_traj(q0, q1, d0, speed=0.5, proid=0.02):
        def interp(s0, s1, d0, num):
            a = d0 + 2*s0 - 2*s1
            b = 3*s1 - 2*d0 - 3*s0
            c = d0
            d = s0
            x = np.linspace(0, 1, num)
            y = ((a*x+b)*x+c)*x+d
            return y
        t = np.linalg.norm(q1 - q0) / speed
        print('exec time is : %.3f' % t)
        inter_num = t // proid + 1
        traj = np.zeros((int(inter_num), 6))
        for i in range(6):
            traj[:, i] = interp(q0[i], q1[i], d0[i], inter_num+1)[1:]
        return traj.tolist()

    def ik(self, m, q_now):
        """ 进行机械臂逆运动学解算,这里解算两次,分别是y轴正和反两个方向 """
        q0 = self.ik_solve(m, q_now)
        return q0
        # mm = m.copy()
        # mm[:3, :2] = -mm[:3, :2]
        # q1 = self.ik_solve(mm, q_now)
        # e0 = np.linalg.norm(q0 - q_now)
        # e1 = np.linalg.norm(q1 - q_now)
        # if e0 < e1:
        #     return q0
        # else:
        #     return q1

    def ik_solve(self, m, q):
        """ 解算器2比较精确但是可能算不出来,这时候使用解算器1 """
        qq = self.ur5_ik.findClosestIK(m, q)
        if qq is None:
            print('解算器2解算失败')
            qq = inv_kin(m, q)
        qs = []
        for q0, qn in zip(qq, q):
            q1 = q0 + np.pi*2 if q0 < 0 else q0 - np.pi * 2
            e0 = abs(q0 - qn)
            e1 = abs(q1 - qn)
            if e0 < e1:
                qs.append(q0)
            else:
                qs.append(q1)
        return qs


class URServo(object):
    """ 开启一个进程用于专门处理机械臂伺服控制 """

    def __init__(self, host, ctrl_period=0.02, speed=0.5):
        self.host = host
        self.pipe, pp = Pipe()
        self.process = URServoProcess(pp, host, ctrl_period, speed)
        self.process.start()

    def __del__(self):
        self.close()

    def __enter__(self):
        print(f"启动机械臂伺服控制进程...")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        print(f"关闭机械臂控制进程...")

    def close(self):
        self.pipe.send(('stop', None))
        self.process.join()

    def servo_to(self, pose):
        # 如果是ur格式的位姿
        if len(pose) == 6:
            pose = ur2np(np.array(pose))
        self.pipe.send(('goto', pose))

    def is_run(self):
        self.pipe.send(('is_run', None))
        _, run = self.pipe.recv()
        return (run > 0)

    def set_speed(self, v):
        self.pipe.send(('set_speed', v))

    def ur_status(self, matrix=True):
        self.pipe.send(('urs', None))
        _, tcp = self.pipe.recv()
        if matrix:
            return ur2np(tcp)
        return tcp

    def ur_qs(self):
        self.pipe.send(('urq', None))
        _, qs = self.pipe.recv()
        return qs

    def clear(self):
        self.pipe.send(('clear', None))

    def movel_to(self, pose, v=0.05):
        # 如果是ur格式的位姿
        if len(pose) != 6:
            pose = np2ur(np.array(pose))
        self.pipe.send(('movel', [pose, v]))
