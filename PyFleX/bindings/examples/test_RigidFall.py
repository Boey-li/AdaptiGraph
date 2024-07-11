import os
import cv2
import argparse
import numpy as np
import pyflex
import time
import torch

from utils import store_data, load_data

np.random.seed(42)


parser = argparse.ArgumentParser()
parser.add_argument('--friction', default=0.1, help='set dynamic friction')
parser.add_argument('--restitution', default=0.1, help='set coefficient of restitution')
parser.add_argument('--g', default=-9.8, help='set Y-axis gravity')
parser.add_argument('--draw_mesh', type=float, default=1, help='visualize particles or mesh')
args = parser.parse_args()


des_dir = 'test_RigidFall'
os.system('mkdir -p ' + des_dir)

time_step = 120
screenWidth = 720
screenHeight = 720
camNear = 0.01
camFar = 1000.


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


pyflex.set_screenWidth(screenWidth)
pyflex.set_screenHeight(screenHeight)
pyflex.init(False)
pyflex.set_light_dir(np.array([0.1, 2.0, 0.1]))
pyflex.set_light_fov(70.)

n_instance = 3
dy_friction = args.friction
restitution = args.restitution
g = args.g

scene_params = np.zeros(n_instance * 3 + 3)
scene_params[0] = n_instance
# scene_params[1] = dy_friction
# scene_params[1] = restitution
scene_params[1] = g
scene_params[-1] = args.draw_mesh

low_bound = 0.09

for i in range(n_instance):
    x = rand_float(0., 0.1)
    y = rand_float(low_bound, low_bound + 0.01)
    z = rand_float(0., 0.1)

    scene_params[i * 3 + 2] = x
    scene_params[i * 3 + 3] = y
    scene_params[i * 3 + 4] = z

    low_bound += 0.21

pyflex.set_scene(3, scene_params, 0)

# front view
pyflex.set_camPos(np.array([0.2, 0.875, 2.0]))
pyflex.set_camAngle(np.array([0., -0.2617994, 0.]))

# left view
# pyflex.set_camPos(np.array([-0.8, 0.875, np.sqrt(3)]))
# pyflex.set_camAngle(np.array([np.radians(-30.), -0.2617994, 0.]))

# right view
# pyflex.set_camPos(np.array([1.1, 0.875, np.sqrt(3)]))
# pyflex.set_camAngle(np.array([np.radians(30.), -0.2617994, 0.]))

# top view
# pyflex.set_camPos(np.array([0.2, 1.2, 1.8]))
# pyflex.set_camAngle(np.array([0., np.radians(-30.), 0.]))

# debug view
# pyflex.set_camPos(np.array([0.2, 0.2, 2.0]))
# pyflex.set_camAngle(np.array([0., 0., 0.]))

print('camPos', pyflex.get_camPos())
print('camAngle', pyflex.get_camAngle())

viewMatrix = pyflex.get_viewMatrix().reshape(4, 4)
projMatrix = pyflex.get_projMatrix().reshape(4, 4)
viewMatrix = np.transpose(viewMatrix)
projMatrix = np.transpose(projMatrix)
print('viewMatrix')
print(viewMatrix)
print('projMatrix')
print(projMatrix)

# pyflex.set_camAngle(np.array([np.radians(45.), -np.radians(20.), 0.]))
# print(pyflex.get_camAngle())

print("Scene Upper:", pyflex.get_scene_upper())
print("Scene Lower:", pyflex.get_scene_lower())

for i in range(time_step):

    positions = pyflex.get_positions().reshape(-1, 4)
    velocities = pyflex.get_velocities().reshape(-1, 3)

    positions[3, 3] = 1.
    uvz = projMatrix.dot(viewMatrix).dot(positions[3])
    uvz = uvz[:3] / uvz[3]
    uvz[0] = (uvz[0] + 1) * screenWidth / 2.
    uvz[1] = (uvz[1] + 1) * screenHeight / 2.
    uvz[2] = uvz[2] * (camFar - camNear) / 2. + (camFar + camNear) / 2.

    print(positions[3], viewMatrix.dot(positions[3]), uvz)

    '''
    store_data(['positions', 'velocities'], [positions[:, :3], velocities[:, :3]],
               path=os.path.join(des_dir, 'info_%d.h5' % (i - 1)))
    '''
    img = pyflex.render(draw_shadow=0).reshape(screenHeight, screenWidth, 4)
    cv2.imwrite(os.path.join(des_dir, 'step_noShadow_%d.png' % i), img[..., :3][..., ::-1])

    img = pyflex.render(draw_shadow=1).reshape(screenHeight, screenWidth, 4)
    cv2.imwrite(os.path.join(des_dir, 'step_%d.png' % i), img[..., :3][..., ::-1])

    pyflex.step()

    if i == 0:
        img = pyflex.render(draw_objects=0).reshape(screenHeight, screenWidth, 4)
        cv2.imwrite(os.path.join(des_dir, 'bg.png'), img[..., :3][..., ::-1])


pyflex.clean()


import cv2
import scipy.misc


fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(os.path.join(des_dir, 'out.avi'), fourcc, 60, (screenWidth, screenHeight))
out_mask = cv2.VideoWriter(os.path.join(des_dir, 'out_mask.avi'), fourcc, 60, (screenWidth, screenHeight))
out_mask_noShadow = cv2.VideoWriter(os.path.join(des_dir, 'out_mask_noShadow.avi'), fourcc, 60, (screenWidth, screenHeight))

bg = cv2.imread(os.path.join(des_dir, 'bg.png'))

for i in range(time_step):
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
