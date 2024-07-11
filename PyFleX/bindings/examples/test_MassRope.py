import os
import pyflex
import time
import torch
import numpy as np

# des_dir = 'test_MassRope'
# os.system('mkdir -p ' + des_dir)


np.random.seed(42)
time_step = 500

screenWidth = 720
screenHeight = 720

def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

def rand_int(lo, hi):
    return np.random.randint(lo, hi)


pyflex.set_screenWidth(screenWidth)
pyflex.set_screenHeight(screenHeight)
pyflex.init(False)

x = 0.
y = 1.0
z = 0.
length = 0.7 # 0.7
stiffness = 1.2
stiffness = rand_float(0.25, 1.2)
draw_mesh = 1.
dt = 1. / 60.

scene_params = np.array([
    x, y, z, length, stiffness, draw_mesh])

pyflex.set_scene(9, scene_params, 0)
pyflex.set_camPos(np.array([0.13, 2.0, 3.2]))

print('camPos:', pyflex.get_camPos())
print("Scene Upper:", pyflex.get_scene_upper())
print("Scene Lower:", pyflex.get_scene_lower())

for i in range(time_step):
    pyflex.step()
pyflex.clean()

# action = np.zeros(3)
# # time.sleep(5)

# imgs = []

# for i in range(time_step):
#     positions = pyflex.get_positions().reshape(-1, 4)

#     # cube: [0, 81]
#     # rope: [81, 96]

#     scale = 0.1
#     action[0] += rand_float(-scale, scale) - positions[-1, 0] * 0.1
#     action[2] += rand_float(-scale, scale) - positions[-1, 2] * 0.1

#     img = pyflex.render().reshape(screenHeight, screenWidth, 4)
#     imgs.append(img[..., :3][..., ::-1])
#     pyflex.step(action * dt)
#     # time.sleep(1.0)

# pyflex.clean()




# import cv2
# import scipy.misc

# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter(os.path.join(des_dir, 'out.avi'), fourcc, 30, (720, 720))
# for i in range(time_step):
#     out.write(imgs[i].astype(np.uint8))
# out.release()
