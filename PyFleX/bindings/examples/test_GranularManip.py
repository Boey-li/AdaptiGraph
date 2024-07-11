import os
import time
import argparse
import cv2
import numpy as np

import pyflex


parser = argparse.ArgumentParser()
parser.add_argument('--cam_idx', type=int, default=0, help='choose from 0 to 20')
parser.add_argument('--draw_mesh', type=int, default=1)

args = parser.parse_args()


des_dir = 'test_CranularManip'
os.system('mkdir -p ' + des_dir)



dt = 1. / 60.

time_step = 120
screenWidth = 720
screenHeight = 720

pyflex.set_screenWidth(screenWidth)
pyflex.set_screenHeight(screenHeight)
pyflex.init(False)



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

print('camPos', pyflex.get_camPos())
print('camAngle', pyflex.get_camAngle())


for r in range(5):
    pusher_angle = np.deg2rad(rand_float(0., 360.))
    pusher_dis = 1.8

    halfEdge = np.array([0.05, 1.0, 0.4])
    center = np.array([
        pusher_dis * np.cos(pusher_angle),
        halfEdge[1],
        pusher_dis * np.sin(pusher_angle)])

    quat = quatFromAxisAngle(
        axis=np.array([0., 1., 0.]),
        angle=-pusher_angle)

    if r == 0:
        pyflex.add_box(halfEdge, center, quat)

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

        pyflex.step()

pyflex.clean()
