import os
import time
import argparse
import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

import pyflex


parser = argparse.ArgumentParser()
parser.add_argument('--cam_idx', type=int, default=0, help='choose from 0 to 20')
parser.add_argument('--draw_mesh', type=int, default=1)

args = parser.parse_args()


des_dir = 'test_GranularManip'
os.system('mkdir -p ' + des_dir)



dt = 1. / 60.

time_step = 120
screenWidth = 720
screenHeight = 720

pyflex.set_screenWidth(screenWidth)
pyflex.set_screenHeight(screenHeight)
pyflex.init()



def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def rand_int(lo, hi):
    return np.random.randint(lo, hi)


def quatFromAxisAngle(axis, angle):
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat


# set scene

radius = 0.075

scale = 1.5
x = - scale / 2.
y = 0.
z = - scale / 2.
staticFriction = 1.0
dynamicFriction = 1.0

for r in range(1):
    scene_params = np.array([
        scale, x, y, z, staticFriction, dynamicFriction])
    pyflex.set_scene(18, scene_params, 0)

    print("Num particles:", pyflex.get_n_particles())


    # set cameras

    cam_idx = args.cam_idx

    rad = np.deg2rad(cam_idx * 20.)
    cam_dis = 7
    cam_height = 4.0
    camPos = np.array([np.sin(rad) * cam_dis, cam_height, np.cos(rad) * cam_dis])
    camAngle = np.array([rad, -np.deg2rad(25.), 0.])

    pyflex.set_camPos(camPos)
    pyflex.set_camAngle(camAngle)
    projMat = pyflex.get_projMatrix()
    viewMat = pyflex.get_viewMatrix()
    projMat = np.array(projMat).reshape(4, 4).T
    viewMat = np.array(viewMat).reshape(4, 4).T
    opencv_T_opengl = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    cx = screenWidth / 2.0
    cy = screenHeight / 2.0
    fx = projMat[0, 0] * cx
    fy = projMat[1, 1] * cy

    print('camPos', pyflex.get_camPos())
    print('camAngle', pyflex.get_camAngle())
    print('projection matrix', projMat)
    print('view matrix', viewMat) # transformation of the world to the camera coordinate system

    # pusher_angle = np.deg2rad(rand_float(0., 360.))
    pusher_angle = 45.
    pusher_dis = 1.8

    halfEdge = np.array([0.05, 1.0, 0.4])
    center = np.array([
        pusher_dis * np.cos(pusher_angle),
        halfEdge[1],
        pusher_dis * np.sin(pusher_angle)])

    quat = quatFromAxisAngle(
        axis=np.array([0., 1., 0.]),
        angle=-pusher_angle)

    hideShape = 0
    color = np.ones(3) * 0.9
    pyflex.add_box(halfEdge, center, quat, hideShape, color)

    # out = cv2.VideoWriter(os.path.join(des_dir, 'comp.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 10, (1200, 500))
    for i in range(time_step):
        shape_states = np.zeros((1, 14))
        shape_states[0, 3:6] = center
        shape_states[0, 6:10] = quat
        shape_states[0, 10:] = quat

        pusher_dis -= 0.015
        center = np.array([
            pusher_dis * np.cos(pusher_angle),
            halfEdge[1],
            pusher_dis * np.sin(pusher_angle)])

        shape_states[0, :3] = center

        pyflex.set_shape_states(shape_states)

        img = pyflex.render(render_depth=True).reshape(screenHeight, screenWidth, 5)
        # cv2.imwrite(os.path.join(des_dir, 'step_%d.png' % i), img[..., :3][..., ::-1])
        # cv2.imwrite(os.path.join(des_dir, 'step_%d_depth.png' % i), (img[:, :, -1]*1000.).astype(np.uint16))

        z = img[:, :, -1] # H, W
        z[z > 10] = 0
        x = (np.tile(np.arange(screenWidth).reshape((1, screenWidth)), (screenHeight, 1)) - cx) * z / fx
        y = (np.tile(np.arange(screenHeight).reshape((screenHeight, 1)), (1, screenWidth)) - cy) * z / fy
        xyz = np.stack([x, y, z], axis=-1) # H, W, 3
        xyz = xyz.reshape(-1, 3) # H*W, 3
        opencv_T_world = np.matmul(np.linalg.inv(viewMat), opencv_T_opengl)
        # print(opencv_T_world)
        xyz = np.matmul(opencv_T_world, np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=-1).T).T[:, :3] # H*W, 3
        # print(xyz)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(img[:, :, :3].reshape(-1, 3) / 255.)
        o3d.io.write_point_cloud(os.path.join(des_dir, 'pcd_%d_cam_%d.ply' % (i, cam_idx)), pcd)

        # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # axes[0].imshow(img[..., :3]/255.)
        # axes[1].imshow(img[..., -1])
        # plt.savefig(os.path.join(des_dir, 'step_%d_comp.png' % i))
        # plt.close()

        # one_comp = cv2.imread(os.path.join(des_dir, 'step_%d_comp.png' % i))
        # out.write(one_comp)

        pyflex.step()
    # out.release()
pyflex.clean()
