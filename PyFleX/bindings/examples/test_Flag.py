import os
import numpy as np
import pyflex
import time
import torch


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

def rand_int(lo, hi):
    return np.random.randint(lo, hi)

def encode(x, y, dimy, offset=0):
    return x * dimy + y + offset


def store_obj(p, dimx, dimy, des_dir, idx):
    filename = os.path.join(des_dir, '%d.obj' % idx)
    fout = open(filename, 'w')

    for i in range(p.shape[0]):
        fout.write('v %.6f %.6f %.6f\n' % (p[i, 0], p[i, 1], p[i, 2]))

    for x in range(dimx - 1):
        for y in range(dimy):
            if y > 0:
                fout.write('f %d %d %d\n' % (
                    encode(x, y, dimy, 1), encode(x, y - 1, dimy, 1), encode(x + 1, y, dimy, 1)))
            if y < dimy - 1:
                fout.write('f %d %d %d\n' % (
                    encode(x, y, dimy, 1), encode(x + 1, y, dimy, 1), encode(x + 1, y + 1, dimy, 1)))

    fout.close()


pyflex.init(False)

time_step = 200
act_scale = 0.2
dimx = 32
dimy = 32
height = 3.0
stiffness = 0.8
stretchStiffness = stiffness
bendStiffness = stiffness
shearStiffness = stiffness
windStrength = 0.0
draw_mesh = 1.

dt = 1. / 60.

scene_params = np.array([
    height, dimx, dimy, stretchStiffness, bendStiffness, shearStiffness,
    windStrength, draw_mesh])

pyflex.set_scene(10, scene_params, 0)

print("Scene Upper:", pyflex.get_scene_upper())
print("Scene Lower:", pyflex.get_scene_lower())

action = np.zeros(3)
update_params = np.zeros(6)
# time.sleep(5)

idx_0, idx_1 = 0, dimx - 1

des_dir = 'Flag'
os.system('mkdir -p ' + des_dir)

pos_rec = np.zeros((time_step, pyflex.get_n_particles(), 4))

for i in range(time_step):
    positions = pyflex.get_positions().reshape(-1, 4)
    pos_rec[i] = positions

    store_obj(positions, dimx, dimy, des_dir, i)

    center = (positions[idx_0] + positions[idx_1]) / 2.

    if i == 0:
        offset = center

    action[0] += rand_float(-act_scale, act_scale) - (center[0] - offset[0]) * act_scale
    action[2] += rand_float(-act_scale, act_scale) - (center[2] - offset[2]) * act_scale

    update_params[0] = action[0]
    update_params[2] = action[2]
    update_params[3] = action[0]
    update_params[5] = action[2]

    # pyflex.step(update_params * dt, capture=1, path=os.path.join(des_dir, 'step_%d.tga' % i))
    pyflex.step()
    # time.sleep(0.03)

    if i == 0:
        time.sleep(1)

time.sleep(1)

pyflex.set_scene(10, scene_params, 0)

for i in range(time_step):
    pyflex.set_positions(pos_rec[i])
    # pyflex.render(capture=0, path=os.path.join(des_dir, 'render_%d.tga' % i))
    pyflex.render()
    # time.sleep(0.03)


pyflex.clean()
