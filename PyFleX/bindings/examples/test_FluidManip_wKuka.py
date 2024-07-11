import argparse
import math
import os
import time
import cv2
import scipy.misc
import pybullet as p

import numpy as np
import pyflex
import torch

from utils import store_data, load_data

from transformations import rotation_matrix, quaternion_from_matrix, quaternion_matrix
from transformations import quaternion_from_euler
from kuka_container import KukaFleXContainer

# np.random.seed(47)

parser = argparse.ArgumentParser()
parser.add_argument('--cam_idx', type=int, default=0, help='choose from 0 to 20')
parser.add_argument('--viscosity', default=0.2, help='set fluid viscosity')
parser.add_argument('--draw_mesh', type=float, default=1, help='visualize particles or mesh')
args = parser.parse_args()

des_dir = 'test_FluidManip'
os.system('mkdir -p ' + des_dir)




dt = 1. / 60.

time_step = 400
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


def quatFromAxisAngle(axis, angle):
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat


def calc_kuka_ee_state(pos, angle, direction, size, border=0.02):
    dx, dy, dz = size
    r_mtx = rotation_matrix(angle, direction)
    quat = quaternion_from_matrix(np.matmul(
        r_mtx,
        quaternion_matrix(quaternion_from_euler(np.pi/2., np.pi/2., np.pi/2.))))

    center = np.array([
        -(dx + border) / 2. - 0.25,
        -(dy + border) / 4.,
        0.,
        1.])
    center = r_mtx.dot(center)[:3] + pos

    return center, quat


def calc_table_shapes(table_size, border, table_height):
    boxes = []

    halfEdge = np.array([table_size / 2., border / 2., table_size / 2.])
    center = np.array([0.4, table_height - border / 2., 0.])
    quat = quaternion_from_euler(0., 0., 0.)

    boxes.append([halfEdge, center, quat])

    return boxes


def calc_container_boxes(pos, angle, direction, size, border=0.02):
    boxes = []
    hide_shape = []

    dx, dy, dz = size
    r_mtx = rotation_matrix(angle, direction)
    quat = quaternion_from_matrix(r_mtx)

    # bottom
    halfEdge = np.array([dx / 2. + border, border / 2., dz / 2. + border])
    center = np.array([0., -(dy + border) / 2., 0., 1.])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(0)

    # left
    halfEdge = np.array([border / 2., dy / 2. + border, dz / 2. + border])
    center = np.array([-(dx + border) / 2., 0., 0., 1.])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(1)

    # right
    center = np.array([(dx + border) / 2., 0., 0., 1.])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(1)

    # back
    halfEdge = np.array([dx / 2. + border, dy / 2. + border, border / 2.])
    center = np.array([0., 0., -(dz + border) / 2., 1.])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(1)

    # front
    center = np.array([0., 0., (dz + border) / 2., 1.])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(1)

    # top bars
    halfEdge = np.array([border / 2., border / 2., dz / 2. + border])
    center = np.array([(dx + border) / 2., (dy + border) / 2., 0., 1])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(0)

    center = np.array([-(dx + border) / 2., (dy + border) / 2., 0., 1])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(0)

    halfEdge = np.array([dx / 2. + border, border / 2., border / 2.])
    center = np.array([0, (dy + border) / 2., (dz + border) / 2., 1])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(0)

    center = np.array([0, (dy + border) / 2., -(dz + border) / 2., 1])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(0)

    # side bars
    halfEdge = np.array([border / 2., dy / 2. + border, border / 2.])
    center = np.array([(dx + border) / 2., 0., (dz + border) / 2., 1])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(0)

    center = np.array([(dx + border) / 2., 0., -(dz + border) / 2., 1])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(0)

    center = np.array([-(dx + border) / 2., 0., -(dz + border) / 2., 1])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(0)

    center = np.array([-(dx + border) / 2., 0., (dz + border) / 2., 1])
    center = r_mtx.dot(center)[:3] + pos
    boxes.append([halfEdge, center, quat])
    hide_shape.append(0)

    return boxes, np.array(hide_shape)


### set scene
# dim_x: 15
# dim_y: 20
# dim_z: 15
table_height = 1.2
table_size = 4.
border = 0.02
radius = 0.055

dim_x_fluid_pourer = 10
dim_y_fluid_pourer = 20
dim_z_fluid_pourer = 10
size_x_pourer = dim_x_fluid_pourer * radius - 0.06
size_y_pourer = 1.2
size_z_pourer = dim_z_fluid_pourer * radius - 0.06

pourer_lim_x = [-0.7, -0.5]
pourer_lim_z = [-0.35, 0.35]
x_pourer = rand_float(pourer_lim_x[0], pourer_lim_x[1] - 0.1)
y_pourer = 1.3 + table_height
z_pourer = rand_float(pourer_lim_z[0], pourer_lim_z[1])
x_fluid_pourer = x_pourer
y_fluid_pourer = y_pourer - size_y_pourer / 2.
z_fluid_pourer = z_pourer


dim_x_fluid_catcher = 25
dim_y_fluid_catcher = 5
dim_z_fluid_catcher = 25
size_x_catcher = dim_x_fluid_catcher * radius - 0.06
size_y_catcher = 0.7
size_z_catcher = dim_z_fluid_catcher * radius - 0.06

x_catcher = 0.4
y_catcher = size_y_catcher / 2. + border + table_height
z_catcher = 0.
x_fluid_catcher = x_catcher
y_fluid_catcher = border + table_height
z_fluid_catcher = z_catcher


draw_mesh = args.draw_mesh

scene_params = np.array([
    x_fluid_pourer - (dim_x_fluid_pourer - 1) / 2. * radius,
    y_fluid_pourer,
    z_fluid_pourer - (dim_z_fluid_pourer - 1) / 2. * radius,
    dim_x_fluid_pourer,
    dim_y_fluid_pourer,
    dim_z_fluid_pourer,
    x_fluid_catcher - (dim_x_fluid_catcher - 1) / 2. * radius,
    y_fluid_catcher,
    z_fluid_catcher - (dim_z_fluid_catcher - 1) / 2. * radius,
    dim_x_fluid_catcher,
    dim_y_fluid_catcher,
    dim_z_fluid_catcher,
    draw_mesh])
print("scene_params", scene_params)
pyflex.set_scene(17, scene_params, 0)

pyflex.set_fluid_color(np.array([0.529, 0.808, 0.98, 0.0]))



# set pourer
pourer_pos = np.array([x_pourer, y_pourer, z_pourer])
pourer_size = np.array([size_x_pourer, size_y_pourer, size_z_pourer])



### set kuka robot

kuka_helper = KukaFleXContainer()
scaling = 4.5
n_link_kuka = 8
rest_pos = [-3.5, 0.14, 0.]
rest_orn = [-0.70710678, 0., 0., 0.70710678]

kuka_helper.add_kuka(rest_pos=rest_pos, rest_orn=rest_orn, scaling=scaling)


for i in range(n_link_kuka):
    pyflex.add_mesh("assets/kuka_iiwa/meshes/link_%d.obj" % i, scaling)

pos, orn = calc_kuka_ee_state(
    pos=pourer_pos,
    angle=0.,
    direction=np.array([0., 0., 1.]),
    size=pourer_size,
    border=border)

# refresh the kuka state history
kuka_helper.set_ee_state(pos=pos, orn=orn)
kuka_shape_states = kuka_helper.get_link_state()

hide_shape_kuka = np.zeros(n_link_kuka)



### set table

boxes = calc_table_shapes(
    table_size, border, table_height)

n_table_shape = len(boxes)

for i in range(n_table_shape):
    halfEdge = boxes[i][0]
    center = boxes[i][1]
    quat = boxes[i][2]
    print(i, halfEdge, center, quat)
    pyflex.add_box(halfEdge, center, quat)

hide_shape_table = np.zeros(n_table_shape)



### set container

boxes_pourer, hide_shape_pourer = calc_container_boxes(
    pos=pourer_pos,
    angle=0.,
    direction=np.array([0., 0., 1.]),
    size=pourer_size,
    border=border)

for i in range(len(boxes_pourer)):
    halfEdge = boxes_pourer[i][0]
    center = boxes_pourer[i][1]
    quat = boxes_pourer[i][2]
    print(i, halfEdge, center, quat)
    pyflex.add_box(halfEdge, center, quat)

# set catcher
catcher_pos = np.array([x_catcher, y_catcher, z_catcher])
catcher_size = np.array([size_x_catcher, size_y_catcher, size_z_catcher])

boxes_catcher, hide_shape_catcher = calc_container_boxes(
    pos=catcher_pos,
    angle=0.,
    direction=np.array([0., 0., 1.]),
    size=catcher_size,
    border=border)

for i in range(len(boxes_catcher)):
    halfEdge = boxes_catcher[i][0]
    center = boxes_catcher[i][1]
    quat = boxes_catcher[i][2]
    print(i, halfEdge, center, quat)
    pyflex.add_box(halfEdge, center, quat)

pyflex.set_hideShapes(np.concatenate(
    [hide_shape_kuka, hide_shape_table, hide_shape_pourer, hide_shape_catcher]))


### set camera
cam_idx = args.cam_idx

rad = np.deg2rad(cam_idx * 18.)
cam_dis = 5.
cam_height = 1.5 + table_height
camPos = np.array([np.sin(rad) * cam_dis, cam_height, np.cos(rad) * cam_dis])
camAngle = np.array([rad, np.deg2rad(0.), 0.])

pyflex.set_camPos(camPos)
pyflex.set_camAngle(camAngle)

print('camPos', pyflex.get_camPos())
print('camAngle', pyflex.get_camAngle())


### read scene info
print("Scene Upper:", pyflex.get_scene_upper())
print("Scene Lower:", pyflex.get_scene_lower())
print("Num particles:", pyflex.get_n_particles())
print("Num shapes:", pyflex.get_n_shapes())



for i in range(2):
    pyflex.step()

for i in range(time_step):

    n_stay_still = 40
    max_angle = 110
    nominal_angle = 65
    if i < n_stay_still:
        angle_cur = 0
        pourer_angle_delta = 0.
        pourer_pos_delta = np.zeros(3)
    else:
        # pourer x position
        scale = 0.002
        pourer_pos_delta[0] += rand_float(-scale, scale) - (pourer_pos[0] - np.sum(pourer_lim_x) / 2.) * scale
        pourer_pos_delta[0] = np.clip(pourer_pos_delta[0], -0.01, 0.01)
        pourer_pos[0] += pourer_pos_delta[0]
        pourer_pos[0] = np.clip(pourer_pos[0], pourer_lim_x[0], pourer_lim_x[1])

        # pourer z position
        scale = 0.003
        pourer_pos_delta[2] += rand_float(-scale, scale) - (pourer_pos[2] - np.sum(pourer_lim_z) / 2.) * scale
        pourer_pos_delta[2] = np.clip(pourer_pos_delta[2], -0.01, 0.01)
        pourer_pos[2] += pourer_pos_delta[2]
        pourer_pos[2] = np.clip(pourer_pos[2], pourer_lim_z[0], pourer_lim_z[1])

        # pourer angle
        scale = 0.3
        angle_idx_cur = i - n_stay_still
        pourer_angle_delta += rand_float(-scale, scale) - (angle_cur - nominal_angle) * 0.002
        pourer_angle_delta = np.clip(pourer_angle_delta, -1.2, 1.2)
        angle_cur += pourer_angle_delta

        # print('delta', pourer_angle_delta)

    pourer_angle = np.deg2rad(max(-angle_cur, -max_angle))

    pourer_prev = boxes_pourer
    boxes_pourer, _ = calc_container_boxes(
        pourer_pos,
        angle=pourer_angle,
        direction=np.array([0., 0., 1.]),
        size=pourer_size,
        border=border)

    catcher_prev = boxes_catcher
    boxes_catcher, _ = calc_container_boxes(
        catcher_pos,
        angle=np.deg2rad(0),
        direction=np.array([0., 0., 1.]),
        size=catcher_size,
        border=border)

    shape_states = np.zeros(
        (n_link_kuka + n_table_shape + len(boxes_pourer) + len(boxes_catcher), 14))


    # set shape state for kuka
    pos, orn = calc_kuka_ee_state(
        pos=pourer_pos,
        angle=pourer_angle,
        direction=np.array([0., 0., 1.]),
        size=pourer_size,
        border=border)

    kuka_helper.set_ee_state(pos=pos, orn=orn)
    shape_states[:n_link_kuka] = kuka_helper.get_link_state()


    # set shape state for the table
    table_shapes = calc_table_shapes(
        table_size, border, table_height)

    offset = n_link_kuka
    for idx_box in range(n_table_shape):
        center = table_shapes[idx_box][1]
        quat = table_shapes[idx_box][2]
        shape_states[idx_box + offset, :3] = center
        shape_states[idx_box + offset, 3:6] = center
        shape_states[idx_box + offset, 6:10] = quat
        shape_states[idx_box + offset, 10:] = quat


    # set shape state for pourer
    offset = n_link_kuka + n_table_shape
    for idx_box in range(len(boxes_pourer)):
        center_prev = pourer_prev[idx_box][1]
        quat_prev = pourer_prev[idx_box][2]
        center = boxes_pourer[idx_box][1]
        quat = boxes_pourer[idx_box][2]

        shape_states[idx_box + offset, :3] = center
        shape_states[idx_box + offset, 3:6] = center_prev
        shape_states[idx_box + offset, 6:10] = quat
        shape_states[idx_box + offset, 10:] = quat_prev

    # set shape state for catcher
    offset = n_link_kuka + n_table_shape + len(boxes_pourer)
    for idx_box in range(len(boxes_catcher)):
        center_prev = catcher_prev[idx_box][1]
        quat_prev = catcher_prev[idx_box][2]
        center = boxes_catcher[idx_box][1]
        quat = boxes_catcher[idx_box][2]

        shape_states[idx_box + offset, :3] = center
        shape_states[idx_box + offset, 3:6] = center_prev
        shape_states[idx_box + offset, 6:10] = quat
        shape_states[idx_box + offset, 10:] = quat_prev


    pyflex.set_shape_states(shape_states)

    img = pyflex.render().reshape(screenHeight, screenWidth, 4)
    cv2.imwrite(os.path.join(des_dir, 'step_%d.png' % i), img[..., :3][..., ::-1])

    pyflex.step()


pyflex.clean()




fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(os.path.join(des_dir, 'out.avi'), fourcc, 60, (screenWidth, screenHeight))

for i in range(time_step):
    # construct mask by comparing with the background
    img = cv2.imread(os.path.join(des_dir, 'step_%d.png' % i))

    # write to video
    out.write(img)

out.release()

