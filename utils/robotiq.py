'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2019-11-09 12:23:59
@LastEditTime : 2020-01-03 20:19:46
@LastEditors  : Lai
'''
import time

import numpy as np
from math import ceil
from .robotiq_gripper import RobotiqGripper


class Robotiq140(RobotiqGripper):
    OPEN_LENGTH = 0.143

    def __init__(self, ip='192.168.1.101', port=63352, timeout=2):
        super().__init__()
        print("Connecting to gripper...")
        self.connect(ip, port, socket_timeout=float(timeout))
        if not self.is_active():
            self.activate()
        else:
            self.auto_calibrate()

    def __del__(self):
        self.disconnect()

    def goto(self, pos, vel=0.1, force=50, wait=False):
        pos_ = int(np.clip((self._min_position-self._max_position) /
                           self.OPEN_LENGTH * pos + self._max_position, 0, 255))
        vel_ = int(np.clip(255./(0.1-0.013) * (vel-0.013), 0, 255))
        force_ = int(np.clip(255./(100.-30.) * (force-30.), 0, 255))
        if not wait:
            return self.move(pos_, vel_, force_)
        else:
            return self.move_and_wait_for_pos(pos_, vel_, force_)
    
    def get_pos(self):
        pos = self.get_current_position()
        return pos



    def open(self, vel=0.08, force=100, wait=True):
        if self.is_open():
            return True
        return self.goto(self.OPEN_LENGTH, vel, force, wait)

    def close(self, vel=0.08, force=100, wait=True):
        if self.is_closed():
            return True
        return self.goto(0, vel, force, wait)
