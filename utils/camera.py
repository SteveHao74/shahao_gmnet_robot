'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2019-10-18 13:49:43
@LastEditTime: 2019-11-27 13:51:15
@LastEditors: Lai
'''
import numpy as np
import pyrender


class CameraBase(object):
    """ 基础的相机类
    相机方程参考https://www.cnblogs.com/ghjnwk/p/10852264.html"""

    def __init__(self, projection_matrix, size):
        """ 一个相机最重要的就是内参矩阵和尺寸
        projection_matrix: 相机投影矩阵,4x4
        size: 相机尺寸(width, height)
        """
        # print(projection_matrix)
        if projection_matrix.shape == (3,3):
            self.matrix = np.eye(4)
            self.matrix[:3,:3] = projection_matrix
        else:
            self.matrix = projection_matrix
        self.size = size
        self.width = self.size[0]
        self.height = self.size[1]
        self.de_matrix = np.linalg.inv(self.matrix)
        print("inverse",self.de_matrix)

    def project(self, point):
        """ 把3d空间中的点投影到图像上
        point: 相机坐标系下的3D点,(x,y,z)
        return: 图像位置和深度, (u,v), d
        """
        assert len(point) == 3, 'the point must has 3 dim'
        p_2d = self.matrix.dot(np.r_[point, 1.])
        d = p_2d[2]
        uv = p_2d[:2] / d
        return uv, d

    def deproject(self, point_image, depth):
        """ 把图像坐标的点投影到相机坐标系
        point_image: 图像坐标下的点, (u,v)
        depth: 该点的深度
        return: 相机坐标系下的3D点,(x,y,z)
        """
        p_2d = depth * np.r_[point_image, 1.]
        
        point_3d = self.de_matrix.dot(np.r_[p_2d, 1.])[:3]
        print("point_3d",point_3d)
        return point_3d


class CameraRT(CameraBase):
    """ 拥有外参数rt的相机类 """

    def __init__(self, projection_matrix, size, rt=None):
        """ rt: 相机的外参数矩阵,世界坐标系到相机坐标系的变换矩阵, 4x4
        """
        super().__init__(projection_matrix, size)
        self.rt = np.eye(4) if rt is None else rt
        self.de_rt = np.linalg.inv(self.rt)

    def project_world(self, point):
        """ 把世界坐标系中的点投影到图像上
        point: 世界坐标系下的3D点,(x,y,z)
        return: 图像位置和深度, (u,v), d
        """
        assert len(point) == 3, 'the point must has 3 dim'
        point_camera = self.rt.dot(np.r_[point, 1.])[:3]
        return self.project(point_camera)

    def deproject_world(self, point_image, depth):
        """ 把图像坐标的点投影到世界坐标系
        point_image: 图像坐标下的点, (u,v)
        depth: 该点的深度
        return: 世界坐标系下的3D点,(x,y,z)
        """
        point_camera = self.deproject(point_image, depth)
        return self.de_rt.dot(np.r_[point_camera, 1.])[:3]


class CameraOpenGLInf(CameraRT):
    """ OpenGL中的无限透视投影相机, 先转换到NDC坐标
    OpenGL相机无法通过单个的投影矩阵表示"""

    def __init__(self, size, pose=None, ndc_matrix=None, model=None):
        """ OpenGL相机模型参考下面两篇博客
        https://blog.csdn.net/linuxheik/article/details/81747087
        https://blog.csdn.net/wangdingqiaoit/article/details/51589825
        https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#projection-matrices
        size: 相机尺寸(width, height)
        pose: 相机在世界坐标系的位姿矩阵,4x4,注意该矩阵的z轴和实际相机坐标系的z轴方向相反
        ndc_matrix: OpenGL无限投影矩阵, 从相机坐标变换到ndc坐标, 4x4
        model: 相机模型, (yfov, aspectRatio, znear), [可视角度, 长宽比例, 焦距(近平面距离)]
        """
        assert ndc_matrix is not None or model is not None, 'must give camera prama'
        self.model = model
        self.ndc_matrix = self.model_to_projection(model) if ndc_matrix is None else ndc_matrix
        self.view_port = self.get_view_port(size)
        projection_matrix = self.view_port.dot(self.ndc_matrix)
        # 这里不设置一下的话矩阵奇异无法求逆
        projection_matrix[-1, -1] = 1
        self.pose = pose
        if self.pose is not None:
            rt = self.pose.copy()
            rt[:3, 2] = -rt[:3, 2]
            rt = np.linalg.inv(rt)
        else:
            rt = None
        super().__init__(projection_matrix, size, rt)

    @staticmethod
    def get_view_port(size):
        """ 获取无限投影矩阵的视口变换矩阵, 从ndc坐标变化到图像坐标
        视口矩阵的推导见doc/OpenGLveiwport.jpg
        其中[-1, 1]要分别映射到[0, width]和[0, height]
        size: 相机尺寸(width, height)
        """
        cx, cy = (np.array(size) - 1) / 2
        # 这里要上下翻转一下所以是-cy
        view_port = np.array([[cx, 0, 0, -cx],
                              [0, -cy, 0, -cy],
                              [0, 0, 0, -1],
                              [0, 0, 0, 1]])
        return view_port

    @staticmethod
    def model_to_projection(model):
        """ 通过相机模型得到OpenGL的ndc无限投影矩阵
        model: 相机模型, [yfov, znear, aspectRatio], [可视角度, 焦距(近平面距离), 长宽比例]
        """
        camera_ = pyrender.PerspectiveCamera(
            yfov=model[0], znear=model[1], aspectRatio=model[2])
        return camera_.get_projection_matrix()
