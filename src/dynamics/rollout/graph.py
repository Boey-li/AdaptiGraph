import os
import glob
import numpy as np
import cv2
import moviepy.editor as mpy
import torch

from dynamics.utils import rgb_colormap, pad, pad_torch
from dynamics.dataset.graph import fps, construct_edges_from_states
from sim.data_gen.data import load_data

def moviepy_merge_video(image_path, image_type, out_path, fps=20):    
    # load images
    image_files = sorted([os.path.join(image_path, img) for img in os.listdir(image_path) if img.endswith(f'{image_type}.jpg')])
    # create a video clip from the images
    clip = mpy.ImageSequenceClip(image_files, fps=fps)
    # write the video clip to a file
    clip.write_videofile(out_path, fps=fps)

def extract_imgs(dataset_config, episode_idx, cam=0):
    ## config
    data_name = dataset_config['data_name']
    data_dir = os.path.join(dataset_config['data_dir'], data_name)
    
    ## load images
    imgs = []
    epi_dir = os.path.join(data_dir, f'{episode_idx:06d}')
    step_list = sorted(list(glob.glob(os.path.join(epi_dir, '*.h5'))))[1:]
    for step_path in step_list:
        data = load_data(step_path)
        color_imgs = data['observations']['color'][f'cam_{cam}'] # (T, H, W, 3)
        imgs.extend(color_imgs)
    imgs = np.array(imgs)
    
    ## load camera
    camera_dir = os.path.join(data_dir, 'cameras')
    cam_extr = np.load(os.path.join(camera_dir, 'extrinsic.npy')) # (n_cameras, 4, 4)
    cam_intr = np.load(os.path.join(camera_dir, 'intrinsic.npy')) # (n_cameras, 4)
    cam_extr, cam_intr = cam_extr[cam], cam_intr[cam] 
    cam_info = {'cam': cam, 'cam_extr': cam_extr, 'cam_intr': cam_intr}
        
    return imgs, cam_info

def visualize_graph(imgs, cam_info, 
        kp_vis, gt_kp_vis, eef_kp, Rr, Rs,
        start, end, vis_t, save_dir, max_nobj,
        colormap=None, point_size=4, edge_size=1, line_size=2, line_alpha=0.5, t_line=5,
        gt_lineset=None, pred_lineset=None, 
        pred_kp_proj_last=None, gt_kp_proj_last=None, physics_param=None):
    
    if colormap is None:
        colormap = rgb_colormap(repeat=100)
    
    if pred_kp_proj_last is None:
        assert gt_kp_proj_last is None
        pred_kp_proj_last = [None]
        gt_kp_proj_last = [None]
    else:
        assert gt_kp_proj_last is not None
    
    pred_kp_proj_list = []
    gt_kp_proj_list = []

    if gt_lineset is None:
        assert pred_lineset is None
        gt_lineset = [[]]
        pred_lineset = [[]]
    else:
        assert pred_lineset is not None
        gt_lineset_new = [[]]
        pred_lineset_new = [[]]
        for lc in range(1):
            for li in range(len(gt_lineset[lc])):
                if gt_lineset[lc][li][-1] >= vis_t - t_line:
                    gt_lineset_new[lc].append(gt_lineset[lc][li])
                    pred_lineset_new[lc].append(pred_lineset[lc][li])
        gt_lineset = gt_lineset_new
        pred_lineset = pred_lineset_new
    
    # image and camera
    intr = cam_info['cam_intr']
    extr = cam_info['cam_extr']
    
    img_orig = imgs[start]
    img = img_orig.copy()

    # transform keypoints
    obj_kp_homo = np.concatenate([kp_vis, np.ones((kp_vis.shape[0], 1))], axis=1) # (N, 4)
    obj_kp_homo = obj_kp_homo @ extr.T  # (N, 4)

    obj_kp_homo[:, 1] *= -1
    obj_kp_homo[:, 2] *= -1

    # project keypoints
    fx, fy, cx, cy = intr
    obj_kp_proj = np.zeros((obj_kp_homo.shape[0], 2))
    obj_kp_proj[:, 0] = obj_kp_homo[:, 0] * fx / obj_kp_homo[:, 2] + cx
    obj_kp_proj[:, 1] = obj_kp_homo[:, 1] * fy / obj_kp_homo[:, 2] + cy
    
    pred_kp_proj_list.append(obj_kp_proj)

    # also transform tool keypoints
    tool_kp_start = eef_kp[0]
    tool_kp_homo = np.concatenate([tool_kp_start, np.ones((tool_kp_start.shape[0], 1))], axis=1) # (N, 4)
    tool_kp_homo = tool_kp_homo @ extr.T  # (N, 4)

    tool_kp_homo[:, 1] *= -1
    tool_kp_homo[:, 2] *= -1

    # also project tool keypoints
    fx, fy, cx, cy = intr
    tool_kp_proj = np.zeros((tool_kp_homo.shape[0], 2))
    tool_kp_proj[:, 0] = tool_kp_homo[:, 0] * fx / tool_kp_homo[:, 2] + cx
    tool_kp_proj[:, 1] = tool_kp_homo[:, 1] * fy / tool_kp_homo[:, 2] + cy

    # visualize
    print(f"shape of obj_kp_proj: {obj_kp_proj.shape}, physics_param shape: {physics_param.shape}")
    # map physics param to color
    phys2color = {}
    count = 0
    for i in range(physics_param.shape[0]):
        if physics_param[i] not in phys2color:
            phys2color[physics_param[i]] = count
            count += 1
    
    for k in range(obj_kp_proj.shape[0]):
        cv2.circle(img, (int(obj_kp_proj[k, 0]), int(obj_kp_proj[k, 1])), point_size, 
            (int(colormap[k, 2]), int(colormap[k, 1]), int(colormap[k, 0])), -1)

    # also visualize tool in red
    for k in range(tool_kp_proj.shape[0]):
        cv2.circle(img, (int(tool_kp_proj[k, 0]), int(tool_kp_proj[k, 1])), 3, 
            (0, 0, 255), -1)
    
    # visualize lineset
    pred_kp_last = pred_kp_proj_last[0]
    if not (pred_kp_last is None):
        for k in range(obj_kp_proj.shape[0]):
            pred_lineset[0].append([int(obj_kp_proj[k, 0]), int(obj_kp_proj[k, 1]), 
                                    int(pred_kp_last[k, 0]), int(pred_kp_last[k, 1]), 
                                    int(colormap[k, 2]), int(colormap[k, 1]), 
                                    int(colormap[k, 0]), vis_t])

    # visualize edges
    for k in range(Rr.shape[0]):
        if Rr[k].sum() == 0: continue
        receiver = Rr[k].argmax()
        sender = Rs[k].argmax()
        if receiver >= max_nobj:  # tool
            cv2.line(img, 
                (int(tool_kp_proj[receiver - max_nobj, 0]), int(tool_kp_proj[receiver - max_nobj, 1])), 
                (int(obj_kp_proj[sender, 0]), int(obj_kp_proj[sender, 1])), 
                (0, 0, 255), edge_size)
        elif sender >= max_nobj:  # tool
            cv2.line(img, 
                (int(tool_kp_proj[sender - max_nobj, 0]), int(tool_kp_proj[sender - max_nobj, 1])), 
                (int(obj_kp_proj[receiver, 0]), int(obj_kp_proj[receiver, 1])), 
                (0, 0, 255), edge_size)
        else:
            cv2.line(img, 
                (int(obj_kp_proj[receiver, 0]), int(obj_kp_proj[receiver, 1])), 
                (int(obj_kp_proj[sender, 0]), int(obj_kp_proj[sender, 1])), 
                (0, 255, 0), edge_size)
    
    # overlay lineset
    img_overlay = img.copy()
    for k in range(len(pred_lineset[0])):
        ln = pred_lineset[0][k]
        cv2.line(img_overlay, (ln[0], ln[1]), (ln[2], ln[3]), (ln[4], ln[5], ln[6]), line_size)

    cv2.addWeighted(img_overlay, line_alpha, img, 1 - line_alpha, 0, img)
    cv2.imwrite(os.path.join(save_dir, f'{start:06}_{end:06}_pred.jpg'), img)
    img_pred = img.copy()

    # visualize gt similarly
    img = img_orig.copy()
    gt_kp_homo = np.concatenate([gt_kp_vis, np.ones((gt_kp_vis.shape[0], 1))], axis=1) # (N, 4)
    gt_kp_homo = gt_kp_homo @ extr.T  # (N, 4)
    gt_kp_homo[:, 1] *= -1
    gt_kp_homo[:, 2] *= -1        
    gt_kp_proj = np.zeros((gt_kp_homo.shape[0], 2))
    gt_kp_proj[:, 0] = gt_kp_homo[:, 0] * fx / gt_kp_homo[:, 2] + cx
    gt_kp_proj[:, 1] = gt_kp_homo[:, 1] * fy / gt_kp_homo[:, 2] + cy

    gt_kp_proj_list.append(gt_kp_proj)
                
    for k in range(gt_kp_proj.shape[0]):
        cv2.circle(img, (int(gt_kp_proj[k, 0]), int(gt_kp_proj[k, 1])), point_size, 
            (int(colormap[k, 2]), int(colormap[k, 1]), int(colormap[k, 0])), -1)

    gt_kp_last = gt_kp_proj_last[0]
    if not (gt_kp_last is None):
        for k in range(gt_kp_proj.shape[0]):
            gt_lineset[0].append([int(gt_kp_proj[k, 0]), int(gt_kp_proj[k, 1]), int(gt_kp_last[k, 0]), int(gt_kp_last[k, 1]), 
                            int(colormap[k, 2]), int(colormap[k, 1]), int(colormap[k, 0]), vis_t])

    # also visualize tool in red
    for k in range(tool_kp_proj.shape[0]):
        cv2.circle(img, (int(tool_kp_proj[k, 0]), int(tool_kp_proj[k, 1])), point_size, 
            (0, 0, 255), -1)

    gt_kp_proj_last.append(obj_kp_proj)

    # visualize edges (for gt, edges will not reflect adjacency)
    for k in range(Rr.shape[0]):
        if Rr[k].sum() == 0: continue
        receiver = Rr[k].argmax()
        sender = Rs[k].argmax()
        if receiver >= max_nobj:  # tool
            cv2.line(img, 
                (int(tool_kp_proj[receiver - max_nobj, 0]), int(tool_kp_proj[receiver - max_nobj, 1])), 
                (int(gt_kp_proj[sender, 0]), int(gt_kp_proj[sender, 1])), 
                (0, 0, 255), edge_size)
        elif sender >= max_nobj:  # tool
            cv2.line(img, 
                (int(tool_kp_proj[sender - max_nobj, 0]), int(tool_kp_proj[sender - max_nobj, 1])), 
                (int(gt_kp_proj[receiver, 0]), int(gt_kp_proj[receiver, 1])), 
                (0, 0, 255), edge_size)
        else:
            cv2.line(img, 
                (int(gt_kp_proj[receiver, 0]), int(gt_kp_proj[receiver, 1])), 
                (int(gt_kp_proj[sender, 0]), int(gt_kp_proj[sender, 1])), 
                (0, 255, 0), edge_size)

    img_overlay = img.copy()
    for k in range(len(gt_lineset[0])):
        ln = gt_lineset[0][k]
        cv2.line(img_overlay, (ln[0], ln[1]), (ln[2], ln[3]), (ln[4], ln[5], ln[6]), line_size)
    
    cv2.imwrite(os.path.join(save_dir, f'{start:06}_{end:06}_gt.jpg'), img)
    img_gt = img.copy()

    img = np.concatenate([img_pred, img_gt], axis=1)
    cv2.imwrite(os.path.join(save_dir, f'{start:06}_{end:06}_both.jpg'), img)
    
    pred_kp_proj_last = pred_kp_proj_list
    gt_kp_proj_last = gt_kp_proj_list
    
    return pred_kp_proj_last, gt_kp_proj_last, gt_lineset, pred_lineset


def construct_graph(dataset_config, material_config, eef_pos, obj_pos,
                    n_his, pair, physics_param, physics_param_shift):
    print("constructing graph for rollout...")
    ## config
    dataset = dataset_config['datasets'][0]
    max_nobj = dataset['max_nobj']
    max_nR = dataset['max_nR']
    fps_radius = (dataset['fps_radius_range'][0] + dataset['fps_radius_range'][1]) / 2
    adj_thresh = (dataset['adj_radius_range'][0] + dataset['adj_radius_range'][1]) / 2
    topk = dataset['topk']
    connect_tool_all = dataset['connect_tool_all']
    
    eef_dim = eef_pos.shape[1]
    obj_dim = max_nobj
    state_dim = obj_dim + eef_dim
    pos_dim = obj_pos.shape[-1]

    ### construct graph ###

    ## get history keypoints
    # obj are probably the object's key points
    # eef are probably the end effector's key points
    obj_kps = []
    eef_kps = []
    for i in range(len(pair)):
        frame_idx = pair[i]
        # eef keypoints
        eef_kp = eef_pos[frame_idx] # (N_eef, 3)
        eef_kps.append(eef_kp)
        # object keypoints
        obj_kp = obj_pos[frame_idx] # (N_obj, 3)
        obj_kps.append(obj_kp)
    
    # obj_kps: (T, N_obj_all, 3)
    # eef_kps: (T, N_eef, 3)
    obj_kps, eef_kps = np.array(obj_kps), np.array(eef_kps)
    
    ## Farthest point sampling for object particles
    # fps_obj_idx: (N_fps, )
    obj_kp_start = obj_kps[n_his-1] # (N_obj, 3)
    fps_idx_list = fps(obj_kp_start, max_nobj, fps_radius)
    obj_kp_num = len(fps_idx_list)

    ## process object keypoints
    # fps_obj_kps: (T, N_obj, 3)
    fps_obj_kp = obj_kps[:, fps_idx_list] # (T, N_fps, 3)
    fps_obj_kps = pad(fps_obj_kp, max_nobj, dim=1) # (T, N_obj, 3)
    
    ## get current state delta (action)
    # states_delta: (N_obj + N_eef, 3)
    eef_kp = np.stack(eef_kps[n_his-1 : n_his+1], axis=0) # (2, N_eef, 3)
    eef_kp_num = eef_kp.shape[1]
    states_delta = np.zeros((state_dim, pos_dim), dtype=np.float32)
    states_delta[obj_dim : obj_dim + eef_kp_num] = eef_kp[1] - eef_kp[0]
    
    ## load history states
    # state_history: (n_his, N_obj + N_eef, 3)
    state_history = np.zeros((n_his, state_dim, pos_dim), dtype=np.float32)
    for fi in range(n_his):
        obj_kp_his = fps_obj_kps[fi] # (N_obj, 3)
        eef_kp_his = eef_kps[fi] # (N_eef, 3)
        state_history[fi] = np.concatenate([obj_kp_his, eef_kp_his], axis=0)
    
    ## load masks
    # state_mask: (N_obj + N_eef, )
    # eef_mask: (N_obj + N_eef, )
    state_mask = np.zeros((state_dim), dtype=bool)
    state_mask[:obj_kp_num] = True
    state_mask[max_nobj : max_nobj + eef_kp_num] = True
    
    eef_mask = np.zeros((state_dim), dtype=bool)
    eef_mask[obj_dim : obj_dim + eef_kp_num] = True
    
    obj_mask = np.zeros((obj_dim), dtype=bool)
    obj_mask[:obj_kp_num] = True

    ## construct attrs
    # attr_dim: (N_obj + N_eef, 2)
    attr_dim = 2 # object + eef
    attrs = np.zeros((state_dim, attr_dim), dtype=np.float32)
    attrs[:obj_kp_num, 0] = 1.
    attrs[max_nobj : max_nobj + eef_kp_num, 1] = 1.
    
    ## construct instance information
    instance_num = 1
    p_rigid = np.zeros(instance_num, dtype=np.float32)
    p_instance = np.zeros((max_nobj, instance_num), dtype=np.float32)
    p_instance[:obj_kp_num, 0] = 1

    # new: construct physics information for each particle
    material_idx = np.zeros((max_nobj, len(material_config['material_index'])), dtype=np.int32)
    assert len(dataset_config['materials']) == 1, 'only support single material'
    material_idx[:obj_kp_num, material_config['material_index'][dataset_config['materials'][0]]] = 1

    # numpy to torch
    state_history = torch.from_numpy(state_history).float()
    states_delta = torch.from_numpy(states_delta).float()
    attrs = torch.from_numpy(attrs).float()
    p_rigid = torch.from_numpy(p_rigid).float()
    p_instance = torch.from_numpy(p_instance).float()
    physics_param = {k: torch.from_numpy(v).float() for k, v in physics_param.items()}
    material_idx = torch.from_numpy(material_idx).long()
    state_mask = torch.from_numpy(state_mask)
    eef_mask = torch.from_numpy(eef_mask)
    obj_mask = torch.from_numpy(obj_mask)
    eef_kp = torch.from_numpy(eef_kp).float()

    # construct relations (density as hyperparameter)
    Rr, Rs = construct_edges_from_states(state_history[-1], adj_thresh, state_mask, eef_mask,
                                         topk, connect_tool_all)
    Rr = pad_torch(Rr, max_nR)
    Rs = pad_torch(Rs, max_nR)

    # save graph
    graph = {
        # input information
        "state": state_history,  # (n_his, N+M, state_dim)
        "action": states_delta,  # (N+M, state_dim)

        # relation information
        "Rr": Rr,  # (n_rel, N+M)
        "Rs": Rs,  # (n_rel, N+M)

        # attr information
        "attrs": attrs,  # (N+M, attr_dim)
        "p_rigid": p_rigid,  # (n_instance,)
        "p_instance": p_instance,  # (N, n_instance)
        # "physics_param": physics_param,  # (N, phys_dim)
        "state_mask": state_mask,  # (N+M,)
        "eef_mask": eef_mask,  # (N+M,)
        "obj_mask": obj_mask,  # (N,)

        "material_index": material_idx,  # (N, num_materials)

        # for non-model use
        "eef_kp": eef_kp,  # (2, N_eef, 3)
    }

    print(f"physics params: {physics_param}")
    print(f"obj dim: {obj_dim}")
    # physics param is normalized between 0 and 1
    mat = None
    for material_name in physics_param.keys():
        #graph[material_name + '_physics_param'] = physics_param[material_name]
        print(f"material: {material_name}, original physics_param: {physics_param[material_name]}")
        mat = material_name
        # Try changing the physics param such that each obj particle has its own physics parameter (N, phys_param)
        # Half normal stiffness, half extra stiffness
        physics_for_each_obj = np.zeros((obj_dim), dtype=np.float32)
        #physics_for_each_obj[:int(obj_dim/2)] = physics_param[material_name].numpy()
        physics_for_each_obj[int(obj_dim/2):] = physics_param[material_name].numpy() + 1.0
        #physics_for_each_obj[:] = physics_param[material_name].numpy() + 1.0
        
        # Alternate chunks of soft and stiff rope
        #step = int(obj_dim/4)
        #physics_for_each_obj[:step] = physics_param[material_name].numpy()
        #physics_for_each_obj[step:2*step] = 0.5  #physics_param[material_name].numpy() + 1.0
        #physics_for_each_obj[2*step:3*step] = physics_param[material_name].numpy() + 1.0
        #physics_for_each_obj[3*step:] = 0.5  #physics_param[material_name] + 1.0
        graph[material_name + "_physics_param"] = torch.tensor(physics_for_each_obj)
        #graph[material_name + "_physics_param"] = physics_param[material_name] + torch.tensor([physics_param_shift])

    print(f"new _physics_param: {graph[mat+'_physics_param']}, size: {graph[mat+'_physics_param'].size()}")
    ### finish constructing graph ###
    print("graph keys: ", graph.keys())
    return graph, fps_idx_list

def get_next_pair_or_break_episode(pairs, n_his, n_frames, current_end):
    # find next pair
    valid_pairs = pairs[pairs[:, n_his-1] == current_end]
    # avoid loop
    valid_pairs = valid_pairs[valid_pairs[:, n_his] > current_end]
    if len(valid_pairs) == 0:
        while current_end < n_frames:
            current_end += 1
            valid_pairs = pairs[pairs[:, n_his-1] == current_end]
            # avoid loop
            valid_pairs = valid_pairs[valid_pairs[:, n_his] > current_end]
            if len(valid_pairs) > 0:
                break
        else:
            return None
    next_pair = valid_pairs[int(len(valid_pairs)/2)]  # pick the middle one
    return next_pair

def get_next_pair_or_break_episode_pushes(pairs, n_his, n_frames, current_end):
    # find next pair
    valid_pairs = pairs[pairs[:, n_his-1] == current_end]
    # avoid loop
    valid_pairs = valid_pairs[valid_pairs[:, n_his] > current_end]
    if len(valid_pairs) == 0:
        return None
    next_pair = valid_pairs[int(len(valid_pairs)/2)]  # pick the middle one
    return next_pair
