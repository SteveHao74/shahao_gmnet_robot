'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2020-01-03 15:05:18
@LastEditTime : 2020-01-17 17:30:02
@LastEditors  : Lai
'''
import os
import cv2
import time
import pyrealsense2 as rs
import numpy as np
from .camera_sensor import CameraSensor


class RealsenseSensor(CameraSensor):
    """ 由于彩色相机和深度相机的视野不一样,不能粗暴的把深度相机对齐到彩色相机 """
    # COLOR_HEIGHT = 720
    # COLOR_WIDTH = 1280
    # DEPTH_HEIGHT = 720
    # DEPTH_WIDTH = 1280
    COLOR_HEIGHT = 720
    COLOR_WIDTH = 1280
    DEPTH_HEIGHT = 768
    DEPTH_WIDTH = 1024
    FPS = 30
    # FILE_PATH = os.path.dirname(os.path.abspath(__file__))
    # INSICS_PATH = os.path.abspath(os.path.join(FILE_PATH, '../cameras/camera_pram/realsense'))

    def __init__(self, rt=None, align_to=None, use='depth', insces_path=None):
        self.size = np.array([self.COLOR_WIDTH, self.COLOR_HEIGHT])
        self.rt = np.load(rt) if isinstance(rt, str) else rt
        self.running = False
        self.align_to = align_to
        self.use = use
        self.insces_path = insces_path

    def __del__(self):
        """Automatically stop the sensor for safety.
        """
        if self.running:
            self.stop()

    def read_insics(self, profile):
        depth_stream = profile.get_stream(rs.stream.depth)
        color_stream = profile.get_stream(rs.stream.color)
        if self.insces_path and os.path.exists(os.path.join(self.insces_path, 'extrinsics.npy')):
            self.extrinsics = np.load(os.path.join(self.insces_path, 'extrinsics.npy'))
        else:
            extr = depth_stream.get_extrinsics_to(color_stream)
            self.extrinsics = np.eye(4)
            self.extrinsics[:3, :3] = np.array(extr.rotation).reshape((3, 3))
            self.extrinsics[:3, 3] = np.array(extr.translation)
        d_intr = depth_stream.as_video_stream_profile().get_intrinsics()
        self.depth_intr = np.array(
            [[d_intr.fx, 0, d_intr.ppx], [0, d_intr.fy, d_intr.ppy], [0, 0, 1]])
        self.depth_dist = np.array(d_intr.coeffs)
        c_intr = color_stream.as_video_stream_profile().get_intrinsics()
        self.color_intr = np.array(
            [[c_intr.fx, 0, c_intr.ppx], [0, c_intr.fy, c_intr.ppy], [0, 0, 1]])
        self.color_dist = np.array(c_intr.coeffs)
        if self.insces_path and os.path.exists(os.path.join(self.insces_path, 'mtx.npy')):
            print('=================================================')
            print('load mtx file', os.path.join(self.insces_path, 'mtx.npy'))
            self.mtx = np.load(os.path.join(self.insces_path, 'mtx.npy'))
        else:
            self.mtx = self.color_intr if self.use == 'color' else self.depth_intr
        if self.insces_path and os.path.exists(os.path.join(self.insces_path, 'dist.npy')):
            print('=================================================')
            print('load dist file', os.path.join(self.insces_path, 'dist.npy'))
            self.dist = np.load(os.path.join(self.insces_path, 'dist.npy'))
        else:
            print('------------------------------------------------')
            self.dist = self.color_dist if self.use == 'color' else self.depth_dist

    def print_camera_info(self):
        print("extrinsics:\n", self.extrinsics)
        print('depth_intr:\n', self.depth_intr)
        print('color_intr:\n', self.color_intr)
        print('depth_dist:\n', self.depth_dist)
        print('color_dist:\n', self.color_dist)
        print('mtx:\n', self.mtx)
        print('dist:\n', self.dist)
        print('image size:', (self.COLOR_HEIGHT, self.COLOR_WIDTH))

    def start(self):
        """ 每次启动相机的内外参都会有微小的变化,所以每次启动时候重新读取内外参 """
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.DEPTH_WIDTH,
                             self.DEPTH_HEIGHT, rs.format.z16, self.FPS)
        config.enable_stream(rs.stream.color, self.COLOR_WIDTH,
                             self.COLOR_HEIGHT, rs.format.rgb8, self.FPS)
        self.profile = self.pipeline.start(config)
        # set exposure time
        color_sensor = self.profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        color_sensor.set_option(rs.option.exposure, 1000)
        # 读取相机参数并重新初始化相机
        self.read_insics(self.profile)
        super().__init__(self.mtx, self.size, self.rt)
        self.print_camera_info()
        if self.align_to is not None:
            if self.align_to == 'color':
                print('-----------------------------align to color')
            self.align = rs.align(rs.stream.color if self.align_to == 'color'
                                  else rs.stream.depth)
        else:
            self.align = None
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.running = True
    
    def get_intrinsics(self):
        depth_stream = self.profile.get_stream(rs.stream.depth)
        color_stream = self.profile.get_stream(rs.stream.color)
        d_intr = depth_stream.as_video_stream_profile().get_intrinsics()
        depth_intr = np.array(
            [[d_intr.fx, 0, d_intr.ppx], [0, d_intr.fy, d_intr.ppy], [0, 0, 1]])
        depth_dist = np.array(d_intr.coeffs)
        c_intr = color_stream.as_video_stream_profile().get_intrinsics()
        color_intr = np.array(
            [[c_intr.fx, 0, c_intr.ppx], [0, c_intr.fy, c_intr.ppy], [0, 0, 1]])
        color_dist = np.array(c_intr.coeffs)
        return depth_intr, depth_dist, color_intr, color_dist


    def stop(self):
        """ Stop the sensor """
        # check that everything is running
        if not self.running:
            return False
        # stop streams
        self.pipeline.stop()
        self.running = False
        return True

    def read(self, timeout=1):
        assert self.running
        frames = self.pipeline.wait_for_frames(timeout_ms=int(timeout*1000))
        if self.align is not None:
            aligned_frames = self.align.process(frames)
        else:
            aligned_frames = frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not color_frame or not aligned_depth_frame:
            False, None, None
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = depth_image * self.depth_scale
        return True, color_image, depth_image

    def frame(self, timeout=1):
        return self.read(timeout)[1:]

    def read_color(self, timeout=1):
        return self.read(timeout)[1]

    def read_depth(self, timeout=1):
        return self.read(timeout)[2]
