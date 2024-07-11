import os
import numpy as np
import pyflex
import time
import torch

import cv2
import scipy.misc


des_dir = 'test_Granular'
os.system('mkdir -p ' + des_dir)

time_step = 400

pyflex.init(False)

scene_params = np.array([])
pyflex.set_scene(13, scene_params, 0)

print("Scene Upper:", pyflex.get_scene_upper())
print("Scene Lower:", pyflex.get_scene_lower())

for i in range(time_step):
    pyflex.step(capture=True, path=os.path.join(des_dir, 'step_%d.tga' % i))

pyflex.clean()


fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(os.path.join(des_dir, 'out.avi'), fourcc, 60, (960, 720))
for i in range(time_step):
    img = scipy.misc.imread(os.path.join(des_dir, 'step_%d.tga' % i))
    out.write(img[..., :3][..., ::-1])
out.release()
