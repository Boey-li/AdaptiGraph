import argparse
import time
import numpy as np
import torch
import open3d as o3d
from PIL import Image
from dgl.geometry import farthest_point_sampler
import os
import yaml

import sys
sys.path.append('.')
from dynamics.utils import fps_rad_idx
from planning.real_world.real_env import RealEnv
from planning.real_world.utils import depth2fgpcd, visualize_o3d

from segment_anything import SamPredictor, sam_model_registry
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

class PerceptionModule:

    def __init__(self, task_config=None, device="cuda:0"):
        if task_config is None:
            self.k_filter = 0.5
            self.obj_list = ['rope']
        else:
            self.k_filter = task_config['k_filter']
            self.obj_list = task_config['obj_list']

        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.device = device
        self.det_model = None
        self.sam_model = None
        self.load_model()

    def load_model(self):
        if self.det_model is not None:
            print("Model already loaded")
            return
        device = self.device
        det_model = build_model(SLConfig.fromfile(
            os.path.join(self.base_path, 'dump/weights/GroundingDINO_SwinB_cfg.py')))
        checkpoint = torch.load(
            os.path.join(self.base_path, 'dump/weights/groundingdino_swinb_cogcoor.pth'), map_location="cpu")
        det_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        det_model.eval()
        det_model = det_model.to(device)

        sam = sam_model_registry["default"](checkpoint=os.path.join(self.base_path, 'dump/weights/sam_vit_h_4b8939.pth'))
        sam_model = SamPredictor(sam)
        sam_model.model = sam_model.model.to(device)

        self.det_model = det_model
        self.sam_model = sam_model
    
    def del_model(self):
        del self.det_model
        torch.cuda.empty_cache()
        del self.sam_model
        torch.cuda.empty_cache()
        self.det_model = None
        self.sam_model = None

    def detect(self, image, captions, box_thresholds):  # captions: list
        image = Image.fromarray(image)
        for i, caption in enumerate(captions):
            caption = caption.lower()
            caption = caption.strip()
            if not caption.endswith("."):
                caption = caption + "."
            captions[i] = caption
        num_captions = len(captions)

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image, None)  # 3, h, w

        image_tensor = image_tensor[None].repeat(num_captions, 1, 1, 1).to(self.device)

        with torch.no_grad():
            outputs = self.det_model(image_tensor, captions=captions)
        logits = outputs["pred_logits"].sigmoid()  # (num_captions, nq, 256)
        boxes = outputs["pred_boxes"]  # (num_captions, nq, 4)

        # filter output
        if isinstance(box_thresholds, list):
            filt_mask = logits.max(dim=2)[0] > torch.tensor(box_thresholds).to(device=self.device, dtype=logits.dtype)[:, None]
        else:
            filt_mask = logits.max(dim=2)[0] > box_thresholds
        labels = torch.ones((*logits.shape[:2], 1)) * torch.arange(logits.shape[0])[:, None, None]  # (num_captions, nq, 1)
        labels = labels.to(device=self.device, dtype=logits.dtype)  # (num_captions, nq, 1)
        logits = logits[filt_mask] # num_filt, 256
        boxes = boxes[filt_mask] # num_filt, 4
        labels = labels[filt_mask].reshape(-1).to(torch.int64) # num_filt,
        scores = logits.max(dim=1)[0] # num_filt,

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {captions[label.item()]} with confidence {round(score.item(), 3)} at location {box}")
        return boxes, scores, labels


    def segment(self, image, boxes, scores, labels, text_prompts):
        # load sam model
        self.sam_model.set_image(image)

        masks, _, _ = self.sam_model.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = self.sam_model.transform.apply_boxes_torch(boxes, image.shape[:2]), # (n_detection, 4)
            multimask_output = False,
        )

        masks = masks[:, 0, :, :] # (n_detection, H, W)
        text_labels = []
        for category in range(len(text_prompts)):
            text_labels = text_labels + [text_prompts[category].rstrip('.')] * (labels == category).sum().item()
        
        # remove masks where IoU are large
        num_masks = masks.shape[0]
        to_remove = []
        for i in range(num_masks):
            for j in range(i+1, num_masks):
                IoU = (masks[i] & masks[j]).sum().item() / (masks[i] | masks[j]).sum().item()
                if IoU > 0.9:
                    if scores[i].item() > scores[j].item():
                        to_remove.append(j)
                    else:
                        to_remove.append(i)
        to_remove = np.unique(to_remove)
        to_keep = np.setdiff1d(np.arange(num_masks), to_remove)
        to_keep = torch.from_numpy(to_keep).to(device=self.device, dtype=torch.int64)
        masks = masks[to_keep]
        text_labels = [text_labels[i] for i in to_keep]
        # text_labels.insert(0, 'background')
        
        aggr_mask = torch.zeros(masks[0].shape).to(device=self.device, dtype=torch.uint8)
        for obj_i in range(masks.shape[0]):
            aggr_mask[masks[obj_i]] = obj_i + 1

        return (masks, aggr_mask, text_labels), (boxes, scores, labels)


    def get_tabletop_points(self, rgb_list, depth_list, R_list, t_list, intr_list, bbox,
                stride=4, depth_threshold=[0, 2], use_raw=False):

        obj_list_full = ['table', 'sheet'] + self.obj_list
        text_prompts = [f"{obj}" for obj in obj_list_full]

        pcd_all = o3d.geometry.PointCloud()
        point_colors = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1]]

        for i in range(len(rgb_list)):
            intr = intr_list[i]
            R_cam2board = R_list[i]
            t_cam2board = t_list[i]

            depth = depth_list[i].copy().astype(np.float32)

            points = depth2fgpcd(depth, intr)
            points = points.reshape(depth.shape[0], depth.shape[1], 3)
            points = points[::stride, ::stride, :].reshape(-1, 3)

            mask = np.logical_and(
                (depth > depth_threshold[0]), (depth < depth_threshold[1])
            )  # (H, W)
            mask = mask[::stride, ::stride].reshape(-1)

            img = rgb_list[i].copy()
            if not use_raw:
                ########## detect and segment ##########
                boxes, scores, labels = self.detect(img, text_prompts, box_thresholds=0.3)

                H, W = img.shape[0], img.shape[1]
                boxes = boxes * torch.Tensor([[W, H, W, H]]).to(device=self.device, dtype=boxes.dtype)
                boxes[:, :2] -= boxes[:, 2:] / 2  # xywh to xyxy
                boxes[:, 2:] += boxes[:, :2]  # xywh to xyxy

                # img = img.astype(np.float32)
                segmentation_results, _ = self.segment(img, boxes, scores, labels, text_prompts)

                masks, _, text_labels = segmentation_results
                masks = masks.detach().cpu().numpy()

                mask_table = np.zeros(masks[0].shape, dtype=np.uint8)
                for obj_i in range(masks.shape[0]):
                    if text_labels[obj_i] == 'table' or text_labels[obj_i] == 'sheet':
                        mask_table = np.logical_or(mask_table, masks[obj_i])

                # add obj_list to mask_table
                mask_obj = np.zeros(masks[0].shape, dtype=np.uint8)
                for text_target in self.obj_list:
                    for obj_i in range(masks.shape[0]):
                        if text_labels[obj_i] == text_target:
                            mask_obj = np.logical_or(mask_obj, masks[obj_i])
                mask_table = np.logical_and(mask_table, ~mask_obj)

                mask_obj_and_background = 1 - mask_table

                mask_obj_and_background = mask_obj_and_background.astype(bool)
                mask_obj_and_background = mask_obj_and_background[::stride, ::stride].reshape(-1)
                mask = np.logical_and(mask, mask_obj_and_background)

            points = points[mask].reshape(-1, 3)

            points = R_cam2board @ points.T + t_cam2board[:, None]
            points = points.T  # (N, 3)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            colors = img[::stride, ::stride, :].reshape(-1, 3).astype(np.float64)
            colors = colors[mask].reshape(-1, 3)
            colors = colors[:, ::-1].copy()
            pcd.colors = o3d.utility.Vector3dVector(colors / 255)

            pcd_all += pcd

        pcd = pcd_all
        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox[:, 0], max_bound=bbox[:, 1]))

        if not use_raw:
            pcd = pcd.voxel_down_sample(voxel_size=0.0005)

            outliers = None
            new_outlier = None
            rm_iter = 0
            while new_outlier is None or len(new_outlier.points) > 0:
                _, inlier_idx = pcd.remove_statistical_outlier(
                    nb_neighbors = 20, std_ratio = 1.5 + rm_iter * 0.5
                )
                new_pcd = pcd.select_by_index(inlier_idx)
                new_outlier = pcd.select_by_index(inlier_idx, invert=True)
                if outliers is None:
                    outliers = new_outlier
                else:
                    outliers += new_outlier
                pcd = new_pcd
                rm_iter += 1

            if self.k_filter < 1.0:
                points = np.array(pcd.points)
                z = points[:, 2]
                z_sorted = np.sort(z)
                z_thresh = z_sorted[int(self.k_filter * len(z_sorted))]
                mask = z < z_thresh
                pcd = pcd.select_by_index(np.where(mask)[0])

        return pcd


def construct_graph(obj_kps, fps_radius, max_nobj=100, max_neef=8, max_nR=500, eef_kps=None, visualize=False):
    if eef_kps is None:
        eef_kps = np.zeros((0, 3))

    if not isinstance(obj_kps, list):
        obj_kps = [obj_kps]

    fps_idx_list = []
    ## sampling using raw particles
    for j in range(len(obj_kps)):
        # farthest point sampling
        particle_tensor = torch.from_numpy(obj_kps[j]).float()[None, ...]
        fps_idx_tensor = farthest_point_sampler(particle_tensor, max_nobj, start_idx=np.random.randint(0, obj_kps[j].shape[0]))[0]
        fps_idx_1 = fps_idx_tensor.numpy().astype(np.int32)

        # downsample to uniform radius
        downsample_particle = particle_tensor[0, fps_idx_1, :].numpy()
        _, fps_idx_2 = fps_rad_idx(downsample_particle, fps_radius)
        fps_idx_2 = fps_idx_2.astype(int)
        fps_idx = fps_idx_1[fps_idx_2]
        fps_idx_list.append(fps_idx)
    
    # downsample to get current obj kp
    obj_kps = [obj_kps[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
    obj_kps = np.concatenate(obj_kps, axis=0) # (N, 3)
    obj_kp_num = obj_kps.shape[0]

    eef_kp_num = eef_kps.shape[0]

    # load masks
    state_mask = np.zeros((max_nobj + max_neef), dtype=bool)
    state_mask[max_nobj : max_nobj + eef_kp_num] = True
    state_mask[:obj_kp_num] = True

    eef_mask = np.zeros((max_nobj + max_neef), dtype=bool)
    eef_mask[max_nobj : max_nobj + eef_kp_num] = True

    state = np.zeros((max_nobj + max_neef, 3))
    state[:obj_kp_num] = obj_kps
    state[max_nobj : max_nobj + eef_kp_num] = eef_kps

    if visualize:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_kps)
        pcd.paint_uniform_color([0, 1, 0])
        pcd_eef = o3d.geometry.PointCloud()
        pcd_eef.points = o3d.utility.Vector3dVector(eef_kps)
        pcd_eef.paint_uniform_color([1, 0, 0])

    state_graph = {
        "obj_state": obj_kps,  # (N, state_dim)
        "obj_state_raw": obj_kps[:obj_kp_num],
        "eef_state": eef_kps,  # (M, state_dim)
        "state": state,  # (N+M, state_dim)
    }

    return state_graph


def get_state_cur(env, pm, device, fps_radius=0.2, visualize=False):
    obs = env.get_obs(get_color=True, get_depth=True)
    intr_list = env.get_intrinsics()
    R_list, t_list = env.get_extrinsics()
    bbox = env.get_bbox()

    rgb_list = []
    depth_list = []
    for i in range(4):
        rgb = obs[f'color_{i}'][-1]
        depth = obs[f'depth_{i}'][-1]
        rgb_list.append(rgb)
        depth_list.append(depth)

    pcd = pm.get_tabletop_points(rgb_list, depth_list, R_list, t_list, intr_list, bbox)
    if visualize:
        visualize_o3d([pcd])
    obj_kps = np.array(pcd.points).astype(np.float32) * env.task_config['sim_real_ratio']
    obj_kps = obj_kps[:, [0, 2, 1]].copy()  # (x, y, z) -> (x, z, y)
    obj_kps[:, 1] *= -1  # (x, z, y) -> (x, -z, y)

    state_graph = construct_graph(obj_kps, fps_radius=fps_radius, visualize=False)
    state_cur = state_graph['obj_state_raw']  # (N, state_dim)
    state_cur = torch.tensor(state_cur, dtype=torch.float32, device=device)

    rgb_vis = obs['color_0'][-1]
    intr = env.get_intrinsics()[0]
    R, t = env.get_extrinsics()[0][0], env.get_extrinsics()[1][0]
    extr = np.eye(4)
    extr[:3, :3] = R.T
    extr[:3, 3] = -R.T @ t
    return state_cur, obj_kps, rgb_vis, intr, extr


def construct_goal_from_perception(task_config):
    base_path = os.path.dirname(os.path.abspath(__file__))

    exposure_time = 5
    env = RealEnv(
        task_config=task_config,
        WH=[1280, 720],
        capture_fps=5,
        obs_fps=5,
        n_obs_steps=1,
        use_robot=True,
        speed=100,
    )

    pm = PerceptionModule(task_config=task_config)

    try:
        env.start(exposure_time=exposure_time)
        env.reset_robot()
        print('env started')
        time.sleep(exposure_time)
        print('start recording')

        env.calibrate(re_calibrate=False)

        obs = env.get_obs(get_color=True, get_depth=True)
        intr_list = env.get_intrinsics()
        R_list, t_list = env.get_extrinsics()
        bbox = env.get_bbox()

        rgb_list = []
        depth_list = []
        for i in range(4):
            rgb = obs[f'color_{i}'][-1]
            depth = obs[f'depth_{i}'][-1]
            rgb_list.append(rgb)
            depth_list.append(depth)
        
        pcd = pm.get_tabletop_points(rgb_list, depth_list, R_list, t_list, intr_list, bbox)

        visualize_o3d([pcd])
        o3d.io.write_point_cloud(os.path.join(base_path, "dump/vis_real_world/target.pcd"), pcd)

    finally:
        env.stop()
        print('env stopped')


def calibrate():

    exposure_time = 5
    env = RealEnv(
        task_config=None,
        WH=[1280, 720],
        capture_fps=5,
        obs_fps=5,
        n_obs_steps=1,
        use_robot=True,
        speed=100
    )
    env.use_hand_eye = True

    try:
        env.start(exposure_time=exposure_time)
        env.reset_robot()
        print('env started')
        time.sleep(exposure_time)
        print('start recording')

        env.calibrate(re_calibrate=True)
    
    finally:
        env.stop()
        print('env stopped')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--construct_goal", action="store_true")
    parser.add_argument("--task_config", type=str, default="")
    args = parser.parse_args()
    if args.calibrate:
        calibrate()
    elif args.construct_goal:
        if args.task_config != "":
            with open(args.task_config, 'r') as f:
                task_config = yaml.load(f, Loader=yaml.CLoader)['task_config']
        else:
            task_config = None
        construct_goal_from_perception(task_config)
    else:
        print("please specify --calibrate or --construct_goal")
