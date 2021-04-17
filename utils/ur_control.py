'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2019-11-05 14:41:08
@LastEditTime : 2020-01-04 20:33:10
@LastEditors  : Lai
'''
import socket
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation


""" 机械臂的IP """
HOST = "192.168.1.101"
""" 要写入的端口 """
PORT = 30003
x = Rotation.from_euler('x', 45, degrees=True).as_dcm()
z = Rotation.from_euler('z', 45, degrees=True).as_dcm()
base2world = np.eye(4)
base2world[:3, :3] = x.dot(z)
base2world[:3, 3] = [0, -0.225, 0.4]
world2base = np.linalg.inv(base2world)


def display(camera_m):
    source_axis = trimesh.creation.axis(origin_color=(255, 0, 0), transform=np.eye(4))
    base_axis = trimesh.creation.axis(origin_color=(0, 0, 255), transform=base2world)
    camera_axis = trimesh.creation.axis(origin_color=(0, 255, 0), transform=camera_m)
    t = np.eye(4)
    t[1, 3] = 0.1
    box = trimesh.creation.box(extents=(0.03, 0.03, 0.03), transform=t)
    scene = trimesh.scene.scene.Scene()
    scene.add_geometry(box)
    scene.add_geometry(base_axis)
    scene.add_geometry(source_axis)
    scene.add_geometry(camera_axis)
    scene.show()


class URControl(object):
    """ 使用socket读取ur5的状态 """

    def __init__(self, host=HOST, port=PORT):
        self.port = port
        self.host = host
        self.is_connect = False
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.settimeout(5)
        self.s.connect((host, port))

    def send(self, byte, timeout=5, host=None, port=None):
        # host = host or self.host
        # port = port or self.port
        # try:
        #     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #     s.settimeout(timeout)
        #     s.connect((host, port))
        #     s.send(byte)
        #     s.close()
        # except socket.timeout:
        #     print("ur连接超时")
        #     return False
        # else:
        #     return True
        self.s.send(byte)

    def movel(self, pose, **arg):
        """ movel指令给定的相对基座的位姿，进行工作中的线性运动 """
        pose_s = str([np.around(p, 4) for p in pose])
        pram = [f'{n}={arg[n]:.3f}' for n in 'a v t r'.split() if n in arg.keys()]
        pram.insert(0, 'p'+pose_s)
        command = f"movel({','.join(pram)})\n"
        command = bytes(command, 'utf-8')
        return self.send(command)

    def movej(self, pose, is_angle=False, **arg):
        """ movej进行关节空间中的线性运动 """
        pose_s = str([np.around(p, 4) for p in pose])
        pram = [f'{n}={arg[n]:.3f}' for n in 'a v t r'.split() if n in arg.keys()]
        pram.insert(0, pose_s if is_angle else ('p' + pose_s))
        command = f"movej({','.join(pram)})\n"
        print(command)
        command = bytes(command, 'utf-8')
        return self.send(command)

    def servoj(self, pose, t=0.008, lookahead=0.1, gain=300):
        """ servoj进行关节空间中的线性运动 """
        assert len(pose) == 6
        pose_s = str([np.around(p, 4) for p in pose])
        lookahead = np.around(np.clip(lookahead, 0.03, 0.2), 4)
        gain = int(np.clip(gain, 100, 2000))
        t = np.around(t, 4)
        # pram = [f'{n}={arg[n]:.3f}' for n in 'a v t r'.split() if n in arg.keys()]
        # pram.insert(0, pose_s)
        command = "servoj({}, 0, 0, {}, {}, {})\n".format(pose_s, t, lookahead, gain)
        # print(co0mmand)
        command = bytes(command, 'utf-8')
        # return command
        return self.send(command)

    def transform_to_base(self, point, angle=0, transform=world2base, show=False):
        # 0.187的时候就碰到了底
        point[2] = point[2] + 0.187
        if point[2] < 0.187:
            print('z is too low, must great than 0.187')
            point[2] = 0.187
        eluer = [0, np.pi, np.deg2rad(angle)+np.pi]
        mat = np.eye(4)
        mat[:3, :3] = Rotation.from_euler('xyz', eluer).as_dcm()
        mat[:3, 3] = point
        if show:
            display(mat)
        mat_base = transform.dot(mat)
        pose = np.zeros((6,))
        pose[:3] = mat_base[:3, 3]
        pose[-3:] = Rotation.from_dcm(mat_base[:3, :3]).as_rotvec()
        return pose
