import os
import sys
import numpy as np
import cv2
from PIL import Image
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', default='FluidIceShake')
parser.add_argument('--st_idx', type=int, default=50)
parser.add_argument('--ed_idx', type=int, default=300)

args = parser.parse_args()

filename = args.src_dir + '.avi'
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter(filename, fourcc, 60, (960, 720))

for i in range(args.st_idx, args.ed_idx):
    img_path = os.path.join(args.src_dir, 'step_%d.tga' % i)
    frame = np.asarray(Image.open(img_path))[:, :, :3]
    out.write(frame[:, :, ::-1])

out.release()

