import os
import numpy as np
import cv2
import torch
import PIL.Image as Image
import moviepy.editor as mpy
import random
from dgl.geometry import farthest_point_sampler

def fps_rad_idx(pcd, radius):
    # pcd: (n, 3) numpy array
    # pcd_fps: (-1, 3) numpy array
    # radius: float
    rand_idx = np.random.randint(pcd.shape[0])
    pcd_fps_lst = [pcd[rand_idx]]
    idx_lst = [rand_idx]
    dist = np.linalg.norm(pcd - pcd_fps_lst[0], axis=1)
    while dist.max() > radius:
        pcd_fps_lst.append(pcd[dist.argmax()])
        idx_lst.append(dist.argmax())
        dist = np.minimum(dist, np.linalg.norm(pcd - pcd_fps_lst[-1], axis=1))
    pcd_fps = np.stack(pcd_fps_lst, axis=0)
    idx_lst = np.stack(idx_lst, axis=0)
    return pcd_fps, idx_lst

def pad(x, max_dim, dim=0):
    if dim == 0:
        x_dim = x.shape[0]
        x_pad = np.zeros((max_dim, x.shape[1]), dtype=np.float32)
        x_pad[:x_dim] = x
    elif dim == 1:
        x_dim = x.shape[1]
        x_pad = np.zeros((x.shape[0], max_dim, x.shape[2]), dtype=np.float32)
        x_pad[:, :x_dim] = x
    return x_pad

def pad_torch(x, max_dim, dim=0):
    if dim == 0:
        x_dim = x.shape[0]
        x_pad = torch.zeros((max_dim, x.shape[1]), dtype=x.dtype, device=x.device)
        x_pad[:x_dim] = x
    elif dim == 1:
        x_dim = x.shape[1]
        x_pad = torch.zeros((x.shape[0], max_dim, x.shape[2]), dtype=x.dtype, device=x.device)
        x_pad[:, :x_dim] = x
    return x_pad

def quaternion_to_rotation_matrix(q):
    # Extract the values from q
    q1, q2, q3, w = q
    
    # First row of the rotation matrix
    r00 = 1 - 2 * (q2 ** 2 + q3 ** 2)
    r01 = 2 * (q1 * q2 - q3 * w)
    r02 = 2 * (q1 * q3 + q2 * w)
    
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q3 * w)
    r11 = 1 - 2 * (q1 ** 2 + q3 ** 2)
    r12 = 2 * (q2 * q3 - q1 * w)
    
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q2 * w)
    r21 = 2 * (q2 * q3 + q1 * w)
    r22 = 1 - 2 * (q1 ** 2 + q2 ** 2)
    
    # Combine all rows into a single matrix
    rotation_matrix = np.array([[r00, r01, r02],
                                [r10, r11, r12],
                                [r20, r21, r22]])
    
    return rotation_matrix

def opencv_merge_video(image_path, image_type, out_path, fps=20):
    f_names = os.listdir(image_path)
    image_names = []
    for f_name in f_names:
        if f'_{image_type}.jpg' in f_name:
            image_names.append(f_name)

    image_names.sort(key=lambda x: int(x.split('_')[0]))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    
    img = Image.open(os.path.join(image_path, image_names[0]))

    video_writer = cv2.VideoWriter(out_path, fourcc, fps, img.size)

    for img_name in image_names:
        img = cv2.imread(os.path.join(image_path, img_name))
        video_writer.write(img)

    # print("Video merged!")

    video_writer.release()

def moviepy_merge_video(image_path, image_type, out_path, fps=20):    
    # load images
    image_files = sorted([os.path.join(image_path, img) for img in os.listdir(image_path) if img.endswith(f'{image_type}.jpg')])
    # create a video clip from the images
    clip = mpy.ImageSequenceClip(image_files, fps=fps)
    # write the video clip to a file
    clip.write_videofile(out_path, fps=fps)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def dataloader_wrapper(dataloader, name):
    cnt = 0
    while True:
        # print(f'[{name}] epoch {cnt}')
        cnt += 1
        for data in dataloader:
            yield data

def grad_manager(phase):
    if phase == 'train':
        return torch.enable_grad()
    else:
        return torch.no_grad()

def truncate_graph(data):
    Rr = data['Rr']
    Rs = data['Rs']
    Rr_nonzero = torch.sum(Rr, dim=-1) > 0
    Rs_nonzero = torch.sum(Rs, dim=-1) > 0
    n_Rr = torch.max(Rr_nonzero.sum(1), dim=0)[0].item()
    n_Rs = torch.max(Rs_nonzero.sum(1), dim=0)[0].item()
    max_n = max(n_Rr, n_Rs)
    data['Rr'] = data['Rr'][:, :max_n, :]
    data['Rs'] = data['Rs'][:, :max_n, :]
    return data

def rgb_colormap(repeat=1):
    base = np.asarray([
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0],
    ])
    return np.repeat(base, repeat, axis=0)