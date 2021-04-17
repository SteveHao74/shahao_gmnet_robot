'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2019-11-05 14:41:08
@LastEditTime : 2019-12-22 13:34:51
@LastEditors  : Lai
'''
import socket
import struct
import numpy as np

""" 机械臂的IP """
HOST = "192.168.1.101"
""" 要读取的端口 """
PORT = 30013


class URStatus(object):
    """ 使用socket读取ur5的状态 """
    # 1.08版本的ur系统是1108字节
    # 1.10版本的是1116字节
    ur_version = 1.08
    Message_size = 1108 if ur_version <= 1.09 else 1116
    Message_fmt = [('Message_Size', 'i'), ('Time', 'd'), ('q_target', '6d'), ('qd_target', '6d'),
                   ('qdd_target', '6d'), ('I_target', '6d'), ('M_target', '6d'), ('q_actual', '6d'),
                   ('qd_actual', '6d'), ('I_actual', '6d'), ('I_control', '6d'),
                   ('Tool_vector_actual', '6d'),
                   ('TCP_speed_actual', '6d'), ('TCP_force', '6d'), ('Tool_vector_target', '6d'),
                   ('TCP_speed_target', '6d'), ('Digital_input_bits', 'd'), ('Motor_temperatures', '6d'),
                   ('Controller_Timer', 'd'), ('Test_value', 'd'), ('Robot_Mode', 'd'),
                   ('Joint_Modes', '6d'), ('Safety_Mode', 'd'), ('empty1', '6d'),
                   ('Tool_Accelerometer_values', '3d'), ('empty2', '6d'), ('Speed_scaling', 'd'),
                   ('Linear_momentum_norm', 'd'), ('SoftwareOnly', 'd'), ('softwareOnly2', 'd'),
                   ('V_main', 'd'), ('V_robot', 'd'), ('I_robot', 'd'), ('V_actual', '6d'),
                   ('Digital_outputs', 'd'), ('Program_state', 'd'), ('Elbow_position', '3d'),
                   ('Elbow_velocity', '3d'), ('Safety_Status', 'd')]
    Message_fmt = Message_fmt[:-1] if ur_version <= 1.09 else Message_fmt

    def __init__(self, host=HOST, port=PORT):
        self.port = port
        self.host = host
        self.is_connect = False
        self.attrs = [f[0] for f in self.Message_fmt]

    def __getattr__(self, attr):
        if attr in self.attrs:
            return self.read(attr)
        else:
            raise AttributeError(attr)

    def read(self, names=None, timeout=5, host=None, port=None):
        """ 这里可能需要加上清空缓冲区的程序 """
        host = host or self.host
        port = port or self.port
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect((host, port))
            data_bytes = s.recv(self.Message_size)
            s.close()
        except socket.timeout:
            print("ur连接超时")
            return None
        else:
            data_dict = self.parse_data(data_bytes)
            if names is None:
                return data_dict
            elif isinstance(names, list):
                result = {n: data_dict.get(n, None) for n in names}
                result = {k: np.array(v) if isinstance(v, list) else v for k, v in result.items()}
                return result
            else:
                result = data_dict.get(names, None)
                return np.array(result) if result else result

    def parse_data(self, data_bytes):
        data_dict = {}
        for (name, fmt) in self.Message_fmt:
            size = struct.calcsize(fmt)
            data = data_bytes[:size]
            data_bytes = data_bytes[size:]
            data_dict[name] = struct.unpack('!'+fmt, data)
        return data_dict
