import argparse
import os
import time
import cv2
import scipy.misc

import numpy as np
import pyflex
import torch

from utils import store_data, load_data, store_data_pickle

np.random.seed(1024)

parser = argparse.ArgumentParser()
parser.add_argument('--cam_idx', type=int, default=0, help='choose from 0 to 20')
parser.add_argument('--draw_mesh', type=float, default=1, help='visualize particles or mesh')
args = parser.parse_args()

des_dir = 'test_SoftFall_SfM'
os.system('mkdir -p ' + des_dir)


dt = 1. / 60.

time_step = 120
screenWidth = 720
screenHeight = 720
camNear = 0.01
camFar = 1000.


pyflex.set_screenWidth(screenWidth)
pyflex.set_screenHeight(screenHeight)
pyflex.init()


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def rand_int(lo, hi):
    return np.random.randint(lo, hi)


### set scene
# type: [0, 1, 2, 3]
# scale: [10, 10]
# x: [0.0, 0.3]
# y: [0.1, 0.2]
# z: [0.0, 0.3]
type_ = 0
scale = 25
x = -1.0 + rand_float(-1., 1.)
y = rand_float(0.2, 2.3)
z = -1.0 + rand_float(-1., 1.)
draw_mesh = args.draw_mesh

y = 2.3

scene_params = np.array([type_, scale, x, y, z, draw_mesh])
print("scene_params", scene_params)
pyflex.set_scene(16, scene_params, 0)


cam_idx = args.cam_idx

# camera pose set 0
rad = np.deg2rad(cam_idx * 18.)
dis = 7.
camPos = np.array([np.sin(rad) * dis, 6., np.cos(rad) * dis])
camAngle = np.array([rad, np.deg2rad(-30.), 0.])


pyflex.set_camPos(camPos)
pyflex.set_camAngle(camAngle)

print('camPos', pyflex.get_camPos())
print('camAngle', pyflex.get_camAngle())



### read scene info
print("Scene Upper:", pyflex.get_scene_upper())
print("Scene Lower:", pyflex.get_scene_lower())
print("Num particles:", pyflex.get_phases().reshape(-1, 1).shape[0])
print("Phases:", np.unique(pyflex.get_phases()))

n_particles = pyflex.get_n_particles()
n_shapes = pyflex.get_n_shapes()
n_rigids = pyflex.get_n_rigids()
n_rigidPositions = pyflex.get_n_rigidPositions()

print("n_particles", n_particles)
print("n_shapes", n_shapes)
print("n_rigids", n_rigids)
print("n_rigidPositions", n_rigidPositions)


for i in range(time_step):

    if i == 80:
        n_views = 300
        viewMatrix = np.zeros((n_views, 4, 4))
        projMatrix = np.zeros((n_views, 4, 4))

        p = pyflex.get_positions().reshape(-1, 4)[:, :3]

        for cam_idx in range(n_views):
            rad = np.deg2rad(cam_idx * (360. / n_views))
            cam_dis = 7.
            cam_height = 6.
            camPos = np.array([np.sin(rad) * cam_dis, cam_height, np.cos(rad) * cam_dis])
            camAngle = np.array([rad, np.deg2rad(-30.), 0.])

            dis = np.sqrt(np.sum((p - camPos) ** 2, 1))
            print('min_dis', np.min(dis), 'max_dis', np.max(dis))

            pyflex.set_camPos(camPos)
            pyflex.set_camAngle(camAngle)

            # print('camPos', pyflex.get_camPos())
            # print('camAngle', pyflex.get_camAngle())

            img = pyflex.render().reshape(screenHeight, screenWidth, 4)
            img = img[..., :3][..., ::-1]
            cv2.imwrite(os.path.join(des_dir, 'view_%d.png' % cam_idx), img)

            bg = pyflex.render(draw_objects=0).reshape(screenHeight, screenWidth, 4)
            bg = bg[..., :3][..., ::-1]
            mask_full = bg != img
            mask = np.logical_or(mask_full[..., 0], mask_full[..., 1])
            mask = np.logical_or(mask, mask_full[..., 2]).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(des_dir, 'mask_%d.png' % cam_idx), mask)


            viewMatrix[cam_idx] = pyflex.get_viewMatrix().reshape(4, 4)
            projMatrix[cam_idx] = pyflex.get_projMatrix().reshape(4, 4)

        data_names = ['viewMatrix', 'projMatrix']
        data = [viewMatrix, projMatrix]
        store_data(data_names, data, os.path.join(des_dir, 'info.h5'))
        store_data_pickle(data_names, data, os.path.join(des_dir, 'info.p'))


    pyflex.step()


pyflex.clean()




'''
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(os.path.join(des_dir, 'out.avi'), fourcc, 60, (screenWidth, screenHeight))
out_mask = cv2.VideoWriter(os.path.join(des_dir, 'out_mask.avi'), fourcc, 60, (screenWidth, screenHeight))
out_mask_noShadow = cv2.VideoWriter(os.path.join(des_dir, 'out_mask_noShadow.avi'), fourcc, 60, (screenWidth, screenHeight))

bg = cv2.imread(os.path.join(des_dir, 'bg.png'))

for i in range(time_step - 1):
    # construct mask by comparing with the background
    img = cv2.imread(os.path.join(des_dir, 'step_%d.png' % i))

    mask_full = bg != img
    mask = np.logical_or(mask_full[..., 0], mask_full[..., 1])
    mask = np.logical_or(mask, mask_full[..., 2]).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(des_dir, 'mask_%d.png' % i), mask)


    # construct mask by comparing with the background
    img_noShadow = cv2.imread(os.path.join(des_dir, 'step_noShadow_%d.png' % i))

    mask_full = bg != img_noShadow
    mask_noShadow = np.logical_or(mask_full[..., 0], mask_full[..., 1])
    mask_noShadow = np.logical_or(mask_noShadow, mask_full[..., 2]).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(des_dir, 'mask_noShadow_%d.png' % i), mask_noShadow)


    # write to video
    out.write(img)
    out_mask.write(np.concatenate([mask[..., None]] * 3, -1))
    out_mask_noShadow.write(np.concatenate([mask_noShadow[..., None]] * 3, -1))


out.release()
out_mask.release()
out_mask_noShadow.release()
'''
