'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2019-11-04 13:14:06
@LastEditTime : 2020-01-03 21:18:31
@LastEditors  : Lai
'''
import os
import cv2
import trimesh
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation
from glob import glob
import numpy as np
file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(file_path, '..'))


x = Rotation.from_euler('x', 45, degrees=True).as_dcm()
z = Rotation.from_euler('z', 45, degrees=True).as_dcm()
base2world = np.eye(4)
base2world[:3, :3] = x.dot(z)
base2world[:3, 3] = [0, -0.225, 0.4]
world2base = np.linalg.inv(base2world)
# print(base2world)


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


def to_matrix(pos):
    r = Rotation.from_rotvec(pos[-3:]).as_dcm()
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = pos[:3]
    return m


def to_vector(m):
    pos = np.zeros((6,))
    pos[-3:] = Rotation.from_dcm(m[:3, :3]).as_rotvec()
    pos[:3] = m[:3, 3]
    return pos


def eye_arm_calibrate(images, urs,  mtx, dist, eye_in_hand=False):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_50)
    parameters = aruco.DetectorParameters_create()
    board = aruco.GridBoard_create(2, 2, 0.08, 0.01, aruco_dict)
    cam = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict,  parameters=parameters)
        retval, rvec, tvec = aruco.estimatePoseBoard(
            corners, ids, board, mtx, dist, None, None)
        cam.append(to_matrix(np.squeeze(np.r_[tvec, rvec])))
    urs = [to_matrix(s) for s in urs]
    if not eye_in_hand:
        urs = [np.linalg.inv(s) for s in urs]
        # cam = [np.linalg.inv(s) for s in cam]
    R_gripper2base = np.array([pos[:3, :3] for pos in urs])
    t_gripper2base = np.array([pos[:3, 3] for pos in urs])
    R_target2cam = np.array([pos[:3, :3] for pos in cam])
    t_target2cam = np.array([pos[:3, 3] for pos in cam])

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam, method=cv2.CALIB_HAND_EYE_PARK)
    cam2base = np.eye(4)
    cam2base[:3, :3] = np.squeeze(R_cam2gripper)
    cam2base[:3, 3] = np.squeeze(t_cam2gripper)
    print(cam2base)
    cam2world = base2world.dot(cam2base)
    print(cam2world)
    zz = cam2world.dot([0, 0, 1, 1])[:3]
    print(zz / np.linalg.norm(zz))
    display(cam2world)
    return cam2base


def calibration_in_file(file_path, mtx_path):
    files = glob(os.path.join(file_path, '*.jpg'))
    mtx = np.load(os.path.join(mtx_path, 'mtx.npy'))
    dist = np.load(os.path.join(mtx_path, 'dist.npy'))
    images = []
    urs = []
    for f in files:
        npy_f = os.path.splitext(f)[0]+'.npy'
        img = cv2.imread(f)
        ur = np.load(npy_f)
        # print(ur)
        urs.append(ur)
        # cv2.imshow('img', img)
        # cv2.waitKey(100)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    eye_arm_calibrate(images, urs, mtx, dist)


def movel_transform(point, transform=world2base, angle=90):
    eluer = [0, np.pi, np.deg2rad(angle)]
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_euler('xyz', eluer).as_dcm()
    mat[:3, 3] = point
    # mat_base = transform.dot(mat)
    # pose = np.zeros((6,))
    # pose[:3] = mat_base[:3, 3]
    # pose[-3:] = Rotation.from_dcm(mat_base[:3, :3]).as_rotvec()
    return mat


if __name__ == "__main__":
    mtx_path = os.path.join(root_path, 'images/primesense')
    out_path = os.path.join(root_path, 'arm_images/primesense')
    calibration_in_file(out_path, mtx_path)
    # 显示末端位置
    # display(movel_transform([0.5, 0, 0.5], angle=0))
    # r = Rotation.from_rotvec([2.2072, -0.9150, -0.3805]).as_dcm()
    # m = np.eye(4)
    # m[:3, :3] = base2world[:3, :3].dot(r)
    # m[:3, 3] = [0.5, 0, 0.5]
    # display(m)