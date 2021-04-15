'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2019-11-04 09:49:30
@LastEditTime : 2020-01-03 16:21:26
@LastEditors  : Lai
'''
from abc import ABCMeta, abstractmethod
from .camera import CameraRT


class CameraSensor(CameraRT, metaclass=ABCMeta):
    """Abstract base class for camera sensors.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def start(self):
        """ start方法用来开启一个相机 """
        pass

    @abstractmethod
    def stop(self):
        """ stop方法用来停止一个相机 """
        pass

    @abstractmethod
    def print_camera_info(self):
        """ 返回表示相机信息的字符串 """
        pass

    def __enter__(self):
        if not self.running:
            self.start()
        print(f"启动相机...")
        self.print_camera_info()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.running:
            self.stop()
        print(f"关闭相机...")

    @abstractmethod
    def frame(self):
        """ 返回最后一帧的图像，同步的RGB和深度图 """
        pass

    def read_depth(self):
        """ 从摄像头读取一张深度图 """
        pass

    def read_color(self):
        """ 从摄像头读取一张rgb图 """
        pass
