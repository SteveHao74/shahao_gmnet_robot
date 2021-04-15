'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2019-12-10 15:58:25
@LastEditTime : 2020-01-17 14:20:01
@LastEditors  : Lai
'''
import numpy as np
import cv2


class ImageProcessor(object):
    """ 用来维护两个图像空间,原始图像与处理后的图像，
    包括生成处理后的图像和两个图像空间的相互转换"""

    def __init__(self, image, width):
        self.original = image
        self.width = width
        self.process, self.zoom, self.crop, self.expend, self.roi = self.process_1(
            self.original, self.width)
        self.image = self.process

    def original_to_process(self, point):
        point = np.array(point)
        zoom = point * self.zoom
        crop = zoom - self.crop
        expend = crop + self.expend
        return expend

    def process_to_original(self, point):
        point = np.array(point)
        expend = point - self.expend
        crop = expend + self.crop
        zoom = crop / self.zoom
        return zoom


    @staticmethod
    def process_1(image, width, out_size=(600, 600), width_des=90):
        image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_DEFAULT)
        mask = ((image > 0.7) | (image < 0.1)).astype(np.uint8)
        depth_scale = np.abs(image).max()
        image = image.astype(np.float32) / depth_scale
        image = cv2.inpaint(image, mask, 1, cv2.INPAINT_NS)
        image = image[10:-10, 10:-10]
        image = image * depth_scale
        # 1.缩放图像,使夹爪宽度符合要求
        zoom_rate = width_des / width
        size_zoom = (int(image.shape[1]*zoom_rate), int(image.shape[0]*zoom_rate))
        image_zoom = cv2.resize(image, size_zoom)
        # 2.如果缩放后的图像比输出大就裁剪一部分
        height, width = image_zoom.shape
        out_width, out_height = out_size
        min_x = max(((width - out_width) // 2), 0)
        min_y = max(((height - out_height) // 2),  0)
        max_x = min(min_x + out_width, width)
        max_y = min(min_y + out_height, height)
        image_crop = image_zoom[min_y:max_y, min_x:max_x]
        # 3.如果裁剪后的图像比输出小,则填补图像
        height, width = image_crop.shape
        start_x = max(((out_width - width) // 2), 0)
        start_y = max(((out_height - height) // 2),  0)
        out_image = np.zeros(out_size[::-1])
        out_image[start_y:start_y+height, start_x:start_x+width] = image_crop
        # 4.修复图像
        out_image = cv2.copyMakeBorder(out_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
        mask = ((out_image > 0.7) | (out_image < 0.1)).astype(np.uint8)
        depth_scale = np.abs(out_image).max()
        out_image = out_image.astype(np.float32) / depth_scale
        out_image = cv2.inpaint(out_image, mask, 1, cv2.INPAINT_NS)
        out_image = out_image[10:-10, 10:-10]
        out_image = out_image * depth_scale
        roi = dict(x_min=start_x, x_max=start_x+width, y_min=start_y, y_max=start_y+height)
        roi = {k: int(v) for k, v in roi.items()}
        return out_image, zoom_rate, np.array([min_x, min_y]), np.array([start_x, start_y]), roi

    def _process(self, image, offset=25, out_size=(600, 600)):
        # 1.得到bin的大概位置, 并裁剪bin内的图像
        # polyContour, hull = self.get_bin(image)
        # max_x, max_y = hull.max(axis=0) - offset
        # min_x, min_y = hull.min(axis=0) + offset * 2
        height, width = image.shape
        out_width, out_height = out_size
        min_x = max(((width - out_width) // 2), 0)
        min_y = max(((height - out_height) // 2),  0)
        max_x = min(min_x + out_width, width)
        max_y = min(min_y + out_height, height)
        image_crop = image[min_y:max_y, min_x:max_x]
        # 1.输入是848*480直接裁出中间区域
        # height, width = image.shape
        # out_width, out_height = out_size
        # if height >out_height:
        #     image = image[(height-out_height) // 2: (height-out_height) // 2+out_height, :]
        # if width >out_width:
        #     image = image[:, (width-out_width) // 2: (width-out_width) // 2+out_width]
        # image_crop = image
        # 2.修复图像,填补空洞
        image_crop = cv2.copyMakeBorder(image_crop, 10, 10, 10, 10, cv2.BORDER_DEFAULT)
        mask = ((image_crop > 0.7) | (image_crop < 0.1)).astype(np.uint8)
        depth_scale = np.abs(image_crop).max()
        image_crop = image_crop.astype(np.float32) / depth_scale
        image_crop = cv2.inpaint(image_crop, mask, 1, cv2.INPAINT_NS)
        image_crop = image_crop[10:-10, 10:-10]
        image_crop = image_crop * depth_scale
        # return image_crop, [0,0 ]
        # 3.在边上插值使图像恢复到300*300
        border_x = int(max(np.ceil((out_size[0] - image_crop.shape[1]) / 2), 0))
        border_y = int(max(np.ceil((out_size[1] - image_crop.shape[0]) / 2), 0))
        h = image_crop.mean()
        # print(h)
        out_image = cv2.copyMakeBorder(image_crop, border_y, border_y,
                                       border_x, border_x, cv2.BORDER_CONSTANT, value=float(h))
        out_image[out_image < 0.1] = h
        crop_x = (out_image.shape[1] - out_size[0])//2
        crop_y = (out_image.shape[0] - out_size[1])//2
        out_image = out_image[int(crop_y):int(crop_y+out_size[1]),
                              int(crop_x):int(crop_x+out_size[0])]
        # 5.计算需要还原到原始图像位置的平移
        x_0 = -min_x+border_x-crop_x
        y_0 = -min_y+border_y-crop_y
        return out_image, np.array([x_0, y_0])

    @staticmethod
    def get_bin(image, thresh=0.03, position='LEFT'):
        """ 得到bin的大概位置 """
        im = image
        im_gray = ((im - im.min()) / (im.max() - im.min()) * 255).astype('uint8')
        im_gray = cv2.medianBlur(im_gray, 7)
        t = thresh * 255 / (np.max(im) - np.min(im))
        im_canny = cv2.Canny(im_gray, t, t*2.5)
        contours, hierarchy = cv2.findContours(im_canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # 面积小于图像面积一般且x轴最大值小于400的轮廓中面积最大一个
        area = np.array([cv2.contourArea(c) for c in contours])
        max_x = np.array([c.max(axis=0)[0, 0] for c in contours])
        position_fiter = (max_x > 400) if position == 'LEFT' else (max_x < 400)
        area = np.where((area > 640*480/2) | (position_fiter), 0, area)
        area_sort = np.argsort(area)[::-1]
        maxa = area_sort[0]
        polyContour = cv2.approxPolyDP(contours[maxa], 20, closed=True)
        hull = cv2.convexHull(polyContour, False)
        return polyContour, np.squeeze(hull)
