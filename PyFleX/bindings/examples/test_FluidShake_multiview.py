import argparse
import os
import time
import cv2
import scipy.misc

import numpy as np
import pyflex
import torch

from utils import store_data, load_data

np.random.seed(1024)

parser = argparse.ArgumentParser()
parser.add_argument('--cam_idx', type=int, default=0, help='choose from 0 to 20')
parser.add_argument('--viscosity', default=0.2, help='set fluid viscosity')
parser.add_argument('--draw_mesh', type=float, default=1, help='visualize particles or mesh')
args = parser.parse_args()

des_dir = 'test_FluidShake_multiview'
os.system('mkdir -p ' + des_dir)


dt = 1. / 60.

time_step = 300
screenWidth = 720
screenHeight = 720
camNear = 0.01
camFar = 1000.

dim_position = 4
dim_velocity = 3
dim_shape_state = 14

border = 0.025
height = 1.3

bar_position_y = 0.6
bar_diameter = 0.04
bar_length_y = 0.4
bar_length_x = 0.2


def calc_box_init(dis_x, dis_z):
    center = np.array([0., 0., 0.])
    quat = np.array([1., 0., 0., 0.])
    boxes = []

    # floor
    halfEdge = np.array([dis_x / 2., border / 2., dis_z / 2.])
    boxes.append([halfEdge, center, quat])

    # left wall
    halfEdge = np.array([border / 2., (height + border) / 2., dis_z / 2.])
    boxes.append([halfEdge, center, quat])

    # right wall
    boxes.append([halfEdge, center, quat])

    # back wall
    halfEdge = np.array([(dis_x + border * 2) / 2., (height + border) / 2., border / 2.])
    boxes.append([halfEdge, center, quat])

    # front wall
    boxes.append([halfEdge, center, quat])


    ## right wall for side bar
    halfEdge = np.array([bar_diameter / 2., (bar_position_y + border) / 2., bar_diameter / 2.])
    boxes.append([halfEdge, center, quat])

    ## right bar
    halfEdge = np.array([bar_length_x / 2., bar_diameter / 2., bar_diameter / 2.])
    boxes.append([halfEdge, center, quat])  # upper side bar
    boxes.append([halfEdge, center, quat])  # lower side bar

    halfEdge = np.array([bar_diameter / 2., bar_length_y / 2., bar_diameter / 2.])
    boxes.append([halfEdge, center, quat])  # middle side bar

    return boxes


def calc_shape_states(x_curr, x_last, z_curr, z_last, box_dis):
    dis_x, dis_z = box_dis
    quat = np.array([1., 0., 0., 0.])

    states = np.zeros((9, dim_shape_state))

    # floor
    states[0, :3] = np.array([x_curr, border / 2., z_curr])
    states[0, 3:6] = np.array([x_last, border / 2., z_last])

    # left wall
    states[1, :3] = np.array([x_curr - (dis_x + border) / 2., (height + border) / 2., z_curr])
    states[1, 3:6] = np.array([x_last - (dis_x + border) / 2., (height + border) / 2., z_last])

    # right wall
    states[2, :3] = np.array([x_curr + (dis_x + border) / 2., (height + border) / 2., z_curr])
    states[2, 3:6] = np.array([x_last + (dis_x + border) / 2., (height + border) / 2., z_last])

    # back wall
    states[3, :3] = np.array([x_curr, (height + border) / 2., z_curr - (dis_z + border) / 2.])
    states[3, 3:6] = np.array([x_last, (height + border) / 2., z_last - (dis_z + border) / 2.])

    # front wall
    states[4, :3] = np.array([x_curr, (height + border) / 2., z_curr + (dis_z + border) / 2.])
    states[4, 3:6] = np.array([x_last, (height + border) / 2., z_last + (dis_z + border) / 2.])

    ## right wall for side bar
    states[5, :3] = np.array([x_curr + (dis_x + border) / 2., (bar_position_y + border) / 2., z_curr])
    states[5, 3:6] = np.array([x_last + (dis_x + border) / 2., (bar_position_y + border) / 2., z_last])

    ## right bar
    states[6, :3] = np.array([x_curr + dis_x / 2. + border + bar_length_x / 2., border + bar_position_y - bar_diameter / 2., z_curr])
    states[6, 3:6] = np.array([x_last + dis_x / 2. + border + bar_length_x / 2., border + bar_position_y - bar_diameter / 2., z_last])
    states[7, :3] = np.array([x_curr + dis_x / 2. + border + bar_length_x / 2., border + bar_position_y - bar_length_y + bar_diameter / 2., z_curr])
    states[7, 3:6] = np.array([x_last + dis_x / 2. + border + bar_length_x / 2., border + bar_position_y - bar_length_y + bar_diameter / 2., z_last])
    states[8, :3] = np.array([x_curr + dis_x / 2. + border + bar_length_x + bar_diameter / 2., border + bar_position_y - bar_length_y / 2., z_curr])
    states[8, 3:6] = np.array([x_last + dis_x / 2. + border + bar_length_x + bar_diameter / 2., border + bar_position_y - bar_length_y / 2., z_last])

    # orientation
    states[:, 6:10] = quat
    states[:, 10:] = quat

    return states


pyflex.set_screenWidth(screenWidth)
pyflex.set_screenHeight(screenHeight)
pyflex.init()


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def rand_int(lo, hi):
    return np.random.randint(lo, hi)


### set scene
# dim_x: [8, 12]
# dim_y: [8, 12]
# dim_z: [4, 6]
# x: [0.0, 0.3]
# y: [0.1, 0.2]
# z: [0.0, 0.3]
dim_x = rand_int(10, 12)
dim_y = rand_int(15, 20)
dim_z = rand_int(10, 12)
x_center = rand_float(-0.2, 0.2)
z_center = rand_float(-0.2, 0.2)
x = x_center - (dim_x - 1) / 2. * 0.055
y = 0.055 / 2. + border + 0.01
z = z_center - (dim_z - 1) / 2. * 0.055
box_dis_x = dim_x * 0.055 + rand_float(0., 0.3)
box_dis_z = dim_z * 0.055 + rand_float(0., 0.3)
draw_mesh = args.draw_mesh

scene_params = np.array([
    x, y, z, dim_x, dim_y, dim_z, box_dis_x, box_dis_z, draw_mesh])
print("scene_params", scene_params)
pyflex.set_scene(6, scene_params, 0)

pyflex.set_fluid_color(np.array([0.529, 0.808, 0.98, 0.0]))

# front view
# pyflex.set_camPos(np.array([0.1, 1.25, 3.]))
# pyflex.set_camAngle(np.array([0., -0.2617994, 0.]))

# left view
# pyflex.set_camPos(np.array([-1.4, 1.25, 1.5 * np.sqrt(3)]))
# pyflex.set_camAngle(np.array([-np.radians(30.), -0.2617994, 0.]))

# right view
# pyflex.set_camPos(np.array([1.6, 1.25, 1.5 * np.sqrt(3)]))
# pyflex.set_camAngle(np.array([np.radians(30.), -0.2617994, 0.]))

# top view
# pyflex.set_camPos(np.array([0.1, 1.6, 2.5]))
# pyflex.set_camAngle(np.array([0., np.radians(-30.), 0.]))


cam_idx = args.cam_idx

rad = np.deg2rad(cam_idx * 18.)
dis = 2.5
camPos = np.array([np.sin(rad) * dis, 1.6, np.cos(rad) * dis])
camAngle = np.array([rad, np.deg2rad(-30.), 0.])

pyflex.set_camPos(camPos)
pyflex.set_camAngle(camAngle)

print('camPos', pyflex.get_camPos())
print('camAngle', pyflex.get_camAngle())


boxes = calc_box_init(box_dis_x, box_dis_z)

for i in range(len(boxes)):
    halfEdge = boxes[i][0]
    center = boxes[i][1]
    quat = boxes[i][2]
    print(i, halfEdge, center, quat)
    pyflex.add_box(halfEdge, center, quat)

hideShapes_off = np.array([0, 1, 1, 1, 1, 0, 0, 1, 1])
hideShapes_on = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
pyflex.set_hideShapes(hideShapes_off)

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

positions = np.zeros((time_step, n_particles, dim_position))
velocities = np.zeros((time_step, n_particles, dim_velocity))
shape_states = np.zeros((time_step, n_shapes, dim_shape_state))

x_box = x_center
z_box = z_center
dx_box = 0
dz_box = 0

for i in range(time_step):
    x_box_last = x_box
    x_box += dx_box * dt
    dx_box += rand_float(-0.1, 0.1) - x_box * 0.1

    z_box_last = z_box
    z_box += dz_box * dt
    dz_box += rand_float(-0.1, 0.1) - z_box * 0.1

    shape_states_ = calc_shape_states(
        x_box, x_box_last, z_box, z_box_last, scene_params[-3:-1])

    pyflex.set_shape_states(shape_states_)

    positions[i] = pyflex.get_positions().reshape(-1, dim_position)
    velocities[i] = pyflex.get_velocities().reshape(-1, dim_velocity)
    shape_states[i] = pyflex.get_shape_states().reshape(-1, dim_shape_state)

    if i == 0:
        print(np.min(positions[i], 0), np.max(positions[i], 0))
        print(x_box, box_dis_x, box_dis_z)

    if i == 0:
        img = pyflex.render(draw_objects=0).reshape(screenHeight, screenWidth, 4)
        cv2.imwrite(os.path.join(des_dir, 'bg.png'), img[..., :3][..., ::-1])
        pyflex.step()
    else:
        '''
        store_data(['positions', 'velocities'], [positions[i, :, :3], velocities[i, :, :3]],
                   path=os.path.join(des_dir, 'info_%d.h5' % (i - 1)))
        '''
        pyflex.set_hideShapes(hideShapes_on)
        img = pyflex.render(draw_shadow=0).reshape(screenHeight, screenWidth, 4)
        cv2.imwrite(os.path.join(des_dir, 'step_noShadow_%d.png' % (i - 1)), img[..., :3][..., ::-1])

        pyflex.set_hideShapes(hideShapes_off)
        img = pyflex.render(draw_shadow=1).reshape(screenHeight, screenWidth, 4)
        cv2.imwrite(os.path.join(des_dir, 'step_%d.png' % (i - 1)), img[..., :3][..., ::-1])

        pyflex.step()


'''
### render

pyflex.set_scene(8, scene_params, 0)
pyflex.set_camPos(np.array([0., 1.25, 3.]))

for i in range(len(boxes) - 1):
    halfEdge = boxes[i][0]
    center = boxes[i][1]
    quat = boxes[i][2]
    pyflex.add_box(halfEdge, center, quat)


for i in range(time_step):
    pyflex.set_positions(positions[i])
    pyflex.set_shape_states(shape_states[i, :-1])

    # pyflex.render(capture=1, path=os.path.join(des_dir, 'render_%d.tga' % i))
    pyflex.render()

    if i == 0:
        time.sleep(1)
'''

pyflex.clean()




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
