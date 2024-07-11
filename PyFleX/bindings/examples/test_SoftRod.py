import os
import numpy as np
import pyflex
import time
import torch


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

def rand_int(lo, hi):
    return np.random.randint(lo, hi)


pyflex.init(False)

dt = 1. / 60.
action_scale = 0.15 * dt
center = np.zeros(2)

y = rand_float(4.5, 7.)
scale = [1.2, y, 1.2]       # x, y, z
trans = [0., 0.1, 0.]       # x, y, z

stiffness = 0.03 + (y - 4) * 0.04
cluster = [2.0, 2.0, stiffness]    # spacing, radius, stiffness
draw_mesh = 1

scene_params = np.array(scale + trans + cluster + [draw_mesh])

pyflex.set_scene(11, scene_params, 0)
print("Scene Upper:", pyflex.get_scene_upper())
print("Scene Lower:", pyflex.get_scene_lower())

action = np.zeros(2)
center = np.zeros(2)
cur = np.zeros(2)

for i in range(300):

    positions = pyflex.get_positions().reshape(-1, 4)
    control_idx = np.arange(positions.shape[0])[positions[:, -1] == 0]

    if i == 0:
        print('# particles', positions.shape[0])
        print('# control', control_idx.shape[0])

    action[0] += rand_float(-action_scale, action_scale) + (center[0] - cur[0]) * action_scale
    action[1] += rand_float(-action_scale, action_scale) + (center[1] - cur[1]) * action_scale
    cur += action

    update_params = np.concatenate([action, control_idx])

    pyflex.step(update_params)

pyflex.clean()

