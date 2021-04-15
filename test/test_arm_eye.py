'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2019-11-11 20:18:42
@LastEditTime: 2019-11-11 22:43:18
@LastEditors: Lai
'''
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

ARM_POS = """0.475732,0.0143899,0.597381,-0.186261,-0.437222,2.36416
0.351412,0.268087,0.458479,0.0520873,-0.0950319,2.38993
0.251188,0.143736,0.426332,-0.216293,0.057463,-0.931251
0.243135,0.151277,0.464429,0.00644091,-0.039015,0.248319
0.288528,0.144912,0.409375,-0.456906,0.134654,-2.23237
0.240534,0.0828331,0.455197,-0.269758,-0.404214,-0.228711
0.358995,0.19536,0.504774,0.226276,0.237398,1.94334
0.188926,0.0555841,0.4517,0.441706,-0.250536,-0.0724471
0.19245,0.143225,0.489355,0.462128,-0.135995,-0.105669
0.379911,0.047085,0.543727,0.361346,-0.141438,3.0347"""
PATTERN_POS = """0.147349 , -0.064831 , 0.509528 , 0.197843 , -0.500065 , 1.792583
-0.106272 , 0.070595 , 0.633095 , 0.078063 , -0.128841 , 1.874043
0.144437 , -0.11512 , 0.602498 , 2.77556 , 2.96201 , 2.070114
-0.042621 , -0.091343 , 0.598773 , 2.95559 , -3.102323 , 0.890111
0.192328 , 0.027207 , 0.552456 , 0.097234 , 0.637267 , 0.328188
0.074878 , -0.10344 , 0.539767 , 2.598053 , 2.800433 , 1.189678
-0.100101 , 0.030685 , 0.526716 , 0.098909 , 0.176179 , 2.323831
0.11135 , -0.055337 , 0.534584 , -2.986191 , 2.769534 , 1.274791
0.018719 , -0.063078 , 0.525738 , -2.923126 , 2.874801 , 1.287745
0.193075 , 0.044841 , 0.52557 , -0.13286 , -0.198836 , 1.280157"""


def to_matrix(pos):
    x = Rotation.from_euler('x', pos[-3]).as_dcm()
    y = Rotation.from_euler('y', pos[-2]).as_dcm()
    z = Rotation.from_euler('z', pos[-1]).as_dcm()
    r = x.dot(y).dot(z)
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = pos[:3]
    return m


def to_vector(m):
    pos = np.zeros((6,))
    pos[-3:] = Rotation.from_dcm(m[:3, :3]).as_rotvec()
    pos[:3] = m[:3, 3]
    return pos


def main():
    arm_pos = ARM_POS.split()
    arm_pos = [np.array(pos.split(','), np.float) for pos in arm_pos]
    arm_pos = np.array(arm_pos)
    pattern_pos = PATTERN_POS.split('\n')
    pattern_pos = [np.array([s.strip() for s in pos.split(',')], np.float) for pos in pattern_pos]
    pattern_pos = np.array(pattern_pos)
    arm_pos = [np.linalg.inv(to_matrix(pos)) for pos in arm_pos]
    # arm_pos = [to_matrix(pos) for pos in arm_pos]
    # pattern_pos = [np.linalg.inv(to_matrix(pos)) for pos in pattern_pos]
    pattern_pos = [to_matrix(pos) for pos in pattern_pos]
    # arm_pos = np.array([to_vector(pos) for pos in arm_pos])
    # pattern_pos = np.array([to_vector(pos) for pos in pattern_pos])
    # print(pattern_pos)
    # fro i in range()
    # R_gripper2base = arm_pos[:, -3:]
    # t_gripper2base = arm_pos[:, :3]
    # R_target2cam = pattern_pos[:, -3:]
    # t_target2cam = pattern_pos[:, :3]
    R_gripper2base = np.array([pos[:3, :3] for pos in arm_pos])
    t_gripper2base = np.array([pos[:3, 3] for pos in arm_pos])
    R_target2cam = np.array([pos[:3, :3] for pos in pattern_pos])
    t_target2cam = np.array([pos[:3, 3] for pos in pattern_pos])
    print(R_target2cam.shape)
    print(t_target2cam.shape)
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam, method=cv2.CALIB_HAND_EYE_PARK)
    print(R_cam2gripper)
    print(t_cam2gripper)


if __name__ == "__main__":
    main()
   
    # print(world2base)
    # print(base2world)
