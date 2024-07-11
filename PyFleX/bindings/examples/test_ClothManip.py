import os
import cv2
import numpy as np
import pyflex
import time
import torch
import scipy.misc


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

def rand_int(lo, hi):
    return np.random.randint(lo, hi)


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def rotate_vector_2d(vector, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(R, vector)


# def render_image_from_PyFleX(pyflex, height, width, debug_info='', draw_objects=1, draw_shadow=1):
#     while True:
#         img = pyflex.render(draw_objects=draw_objects, draw_shadow=draw_shadow)
#         img = img.reshape(height, width, 4)
#         img = img[..., :3][..., ::-1]
#         if (img != 0).any():
#             break
#         else:
#             print('empty image at %s' % debug_info)
#     return img



pyflex.init(False)

time_step = 300
dt = 1. / 60.

# offset
radius = 0.05
offset_x = -1.
offset_y = 0.06
offset_z = -1.

# fabrics
fabric_type = rand_int(0, 3) # 0: Cloth, 1: shirt, 2: pants

if fabric_type == 0:
    # parameters of the shape
    dimx = rand_int(25, 35)    # dimx, width
    dimy = rand_int(25, 35)    # dimy, height
    dimz = 0
    # the actuated points
    ctrl_idx = np.array([
        0, dimx // 2, dimx - 1,
        dimy // 2 * dimx,
        dimy // 2 * dimx + dimx - 1,
        (dimy - 1) * dimx,
        (dimy - 1) * dimx + dimx // 2,
        (dimy - 1) * dimx + dimx - 1])

    offset_x = -dimx * radius / 2.
    offset_y = 0.06
    offset_z = -dimy * radius / 2.

elif fabric_type == 1:
    # parameters of the shape
    dimx = rand_int(16, 25)     # width of the body
    dimy = rand_int(30, 35)     # height of the body
    dimz = 7                    # size of the sleeves
    # the actuated points
    ctrl_idx = np.array([
        dimx * dimy,
        dimx * dimy + dimz * (dimz + dimz // 2) + (1 + dimz) * (dimz + 1) // 4,
        dimx * dimy + (1 + dimz) * dimz // 2 + dimz * (dimz - 1),
        dimx * dimy + dimz * (dimz + dimz // 2) + (1 + dimz) * (dimz + 1) // 4 + \
            (1 + dimz) * dimz // 2 + dimz * dimz - 1,
        dimy // 2 * dimx,
        dimy // 2 * dimx + dimx - 1,
        (dimy - 1) * dimx,
        dimy * dimx - 1])

    offset_x = -(dimx + dimz * 4) * radius / 2.
    offset_y = 0.06
    offset_z = -dimy * radius / 2.

elif fabric_type == 2:
    # parameters of the shape
    dimx = rand_int(9, 13) * 2 # width of the pants
    dimy = rand_int(6, 11)      # height of the top part
    dimz = rand_int(24, 31)     # height of the leg
    # the actuated points
    ctrl_idx = np.array([
        0, dimx - 1,
        (dimy - 1) * dimx,
        (dimy - 1) * dimx + dimx - 1,
        dimx * dimy + dimz // 2 * (dimx - 4) // 2,
        dimx * dimy + (dimz - 1) * (dimx - 4) // 2,
        dimx * dimy + dimz * (dimx - 4) // 2 + 3 + \
            dimz // 2 * (dimx - 4) // 2 + (dimx - 4) // 2 - 1,
        dimx * dimy + dimz * (dimx - 4) // 2 + 3 + \
            dimz * (dimx - 4) // 2 - 1])

    offset_x = -dimx * radius / 2.
    offset_y = 0.06
    offset_z = -(dimy + dimz) * radius / 2.


# physical param
stiffness = rand_float(0.4, 1.0)
stretchStiffness = stiffness
bendStiffness = stiffness
shearStiffness = stiffness

dynamicFriction = 0.6
staticFriction = 1.0
particleFriction = 0.6

invMass = 1.0

# other parameters
windStrength = 0.0
draw_mesh = 1.

# set up environment
scene_params = np.array([
    offset_x, offset_y, offset_z,
    fabric_type, dimx, dimy, dimz,
    ctrl_idx[0], ctrl_idx[1], ctrl_idx[2], ctrl_idx[3],
    ctrl_idx[4], ctrl_idx[5], ctrl_idx[6], ctrl_idx[7],
    stretchStiffness, bendStiffness, shearStiffness,
    dynamicFriction, staticFriction, particleFriction,
    invMass, windStrength, draw_mesh])

scene_idx = 15
pyflex.set_scene(scene_idx, scene_params, 0)

# set up camera pose
camPos = np.array([0., 3.5, 0.])
camAngle = np.array([0., -90./180. * np.pi, 0.])
pyflex.set_camPos(camPos)
pyflex.set_camAngle(camAngle)


# let the cloth drop
action = np.zeros(4)
for i in range(5):
    pyflex.step(action)



# des_dir = 'ClothManip'
# os.system('mkdir -p ' + des_dir)

pos_rec = np.zeros((time_step, pyflex.get_n_particles(), 3))

print('n_particles', pyflex.get_n_particles())

imgs = []
for i in range(time_step):
    positions = pyflex.get_positions().reshape(-1, 4)[:, :3]
    pos_rec[i] = positions

    center = np.sum(positions[ctrl_idx], 0) / ctrl_idx.shape[0]

    if i % 5 == 0:
        ctrl_pts = rand_int(0, 8)

        '''
        act_lim = [0.015, 0.03]
        theta_lim = [np.deg2rad(-90.), np.deg2rad(90)]

        theta = rand_float(theta_lim[0], theta_lim[1])
        scale = rand_float(act_lim[0], act_lim[1])

        p = positions[ctrl_idx[ctrl_pts]]
        direction = unit_vector(p - center)[[0, 2]]
        direction = rotate_vector_2d(direction, theta)

        dx = direction[0] * scale
        dz = direction[1] * scale
        '''

        act_lim = 0.05
        dx = rand_float(-act_lim, act_lim)
        dz = rand_float(-act_lim, act_lim)
        dy = 0.05

        action = np.array([ctrl_pts, dx, dy, dz])

    else:
        action[2] = 0.

    # img = render_image_from_PyFleX(pyflex, height=720, width=960)
    # imgs.append(img)

    pyflex.step(action)


for i in range(100):
    # img = render_image_from_PyFleX(pyflex, height=720, width=960)
    # imgs.append(img)

    action = np.zeros(4)
    pyflex.step(action)

pyflex.clean()


'''
pyflex.set_scene(scene_idx, scene_params, 0)

pyflex.set_camPos(camPos)
pyflex.set_camAngle(camAngle)

for i in range(time_step):
    pyflex.set_positions(pos_rec[i])
    pyflex.render(capture=0, path=os.path.join(des_dir, 'render_%d.tga' % i))
    # time.sleep(0.03)
'''

# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter(os.path.join(des_dir, 'out.avi'), fourcc, 40, (960, 720))
# for i in range(time_step):
#     out.write(imgs[i].astype(np.uint8))
# out.release()
