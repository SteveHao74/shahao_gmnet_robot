'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2019-11-04 10:06:58
@LastEditTime : 2020-01-16 15:17:38
@LastEditors  : Lai
'''
import os
import numpy as np
from primesense import openni2
from .camera_sensor import CameraSensor


class PrimesenseSensor(CameraSensor):
    """ Primesense1.08或者1.09相机类
    """
    # Constants for image height and width (in case they're needed somewhere)
    COLOR_IM_HEIGHT = 480
    COLOR_IM_WIDTH = 640
    DEPTH_IM_HEIGHT = 480
    DEPTH_IM_WIDTH = 640
    CENTER_X = float(DEPTH_IM_WIDTH-1) / 2.0
    CENTER_Y = float(DEPTH_IM_HEIGHT-1) / 2.0
    FOCAL_X = 525.
    FOCAL_Y = 525.
    FPS = 30
    OPENNI2_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Redist')

    def __init__(self, auto_white_balance=False, auto_exposure=True,
                 enable_depth_color_sync=True, flip_images=False, insces_path=None):
        self.device = None
        self.depth_stream = None
        self.color_stream = None
        self.ir_stream = None
        self.running = None
        self.width = self.COLOR_IM_WIDTH
        self.height = self.COLOR_IM_HEIGHT

        self.auto_white_balance = auto_white_balance
        self.auto_exposure = auto_exposure
        self.enable_depth_color_sync = enable_depth_color_sync
        self.flip_images = flip_images
        if insces_path is not None:
            self.mtx = np.load(os.path.join(insces_path, 'mtx.npy'))
            self.dist = np.load(os.path.join(insces_path, 'dist.npy'))
            super().__init__(self.mtx, [self.COLOR_IM_WIDTH, self.COLOR_IM_HEIGHT])

    def __del__(self):
        """Automatically stop the sensor for safety.
        """
        if self.running:
            self.stop()

    def start(self):
        """ Start the sensor """
        # 初始化openni
        openni2.initialize(self.OPENNI2_PATH)
        self.device = openni2.Device.open_any()

        # 开启相机深度图流
        self.depth_stream = self.device.create_depth_stream()
        self.depth_stream.configure_mode(self.DEPTH_IM_WIDTH,
                                         self.DEPTH_IM_HEIGHT,
                                         self.FPS,
                                         openni2.PIXEL_FORMAT_DEPTH_1_MM)
        self.depth_stream.start()

        # 开启相机彩色图流
        self.color_stream = self.device.create_color_stream()
        self.color_stream.configure_mode(self.COLOR_IM_WIDTH,
                                         self.COLOR_IM_HEIGHT,
                                         self.FPS,
                                         openni2.PIXEL_FORMAT_RGB888)
        self.color_stream.camera.set_auto_white_balance(self.auto_white_balance)
        self.color_stream.camera.set_auto_exposure(self.auto_exposure)
        self.color_stream.start()

        # 把深度相机和rgb相机对齐，参考https://kheresy.wordpress.com/2013/01/04/videostream-and-device-of-openni2/
        self.device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        self.device.set_depth_color_sync_enabled(self.enable_depth_color_sync)

        self.running = True

    def stop(self):
        """ Stop the sensor """
        # check that everything is running
        if not self.running or self.device is None:
            return False

        # stop streams
        if self.depth_stream:
            self.depth_stream.stop()
        if self.color_stream:
            self.color_stream.stop()
        self.running = False

        openni2.unload()
        return True

    def _read_depth(self):
        """ Reads a depth image from the device """
        # s = openni2.wait_for_any_stream([self.depth_stream], timeout)
        # if not s:
        #     return None
        # read raw uint16 buffer
        im_arr = self.depth_stream.read_frame()
        raw_buf = im_arr.get_buffer_as_uint16()
        depth_im = np.ctypeslib.as_array(raw_buf).reshape(
            (self.DEPTH_IM_HEIGHT, self.DEPTH_IM_WIDTH))
        depth_im = depth_im.astype('float64') / 1000  # 把毫米的单位转换到米
        # TODO: 不是很清楚为什么要翻转图像
        if self.flip_images:
            # 上下翻转图像
            depth_im = np.flipud(depth_im)
        else:
            # 左右翻转图像
            depth_im = np.fliplr(depth_im)
        return depth_im

    def _read_color(self):
        """ Reads a color image from the device
        这里输出的时RGB格式的图，在opencv处理时需要转换一下 """
        # s = openni2.wait_for_any_stream([self.color_stream], timeout)
        # if not s:
        #     return None
        # read raw buffer
        im_arr = self.color_stream.read_frame()
        raw_buf = im_arr.get_buffer_as_triplet()
        color_im = np.ctypeslib.as_array(raw_buf)
        color_im = color_im.reshape((self.COLOR_IM_HEIGHT, self.COLOR_IM_WIDTH, 3))
        if self.flip_images:
            color_im = np.flipud(color_im.astype(np.uint8))
        else:
            color_im = np.fliplr(color_im.astype(np.uint8))
        return color_im

    def read(self, timeout=2):
        # 这里一定要同时读取深度和rgb图,只读一个会一直在那里等待
        s = openni2.wait_for_any_stream([self.color_stream, self.depth_stream], timeout)
        if not s:
            return False, None, None
        color_im = self._read_color()
        depth_im = self._read_depth()
        return True, color_im, depth_im

    def frame(self, timeout=2):
        return self.read(timeout)[1:]

    def read_color(self, timeout=2):
        return self.read(timeout)[1]

    def read_depth(self, timeout=2):
        return self.read(timeout)[2]

    def print_camera_info(self):
        print(self.device.get_device_info())
        print("v-fov:", self.depth_stream.get_vertical_fov())
        print("h-fov:", self.depth_stream.get_horizontal_fov())
        print("camera settings:", self.depth_stream.camera)
        print('color image size:', (self.COLOR_IM_HEIGHT, self.COLOR_IM_WIDTH, 3))
        print('depth image size:', (self.DEPTH_IM_HEIGHT, self.DEPTH_IM_WIDTH))
