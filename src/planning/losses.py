import torch
import numpy as np

def chamfer(x, y):  # x: (B, N, D), y: (B, M, D)
    x = x[:, None].repeat(1, y.shape[1], 1, 1)  # (B, M, N, D)
    y = y[:, :, None].repeat(1, 1, x.shape[2], 1)  # (B, M, N, D)
    dis = torch.norm(x - y, 2, dim=-1)  # (B, M, N)
    dis_xy = torch.mean(dis.min(dim=2).values, dim=1)  # dis_xy: mean over N
    dis_yx = torch.mean(dis.min(dim=1).values, dim=1)  # dis_yx: mean over M
    return dis_xy + dis_yx

def mean_chamfer(state_pred, state_real, state_pred_mask, state_real_mask):
    # chamfer distance
    # state_pred: numpy.ndarray (bsz, max_nobj, 3)
    # state_real: numpy.ndarray (bsz, max_nobj, 3)
    bsz = state_pred.shape[0]
    chamfer_dist_list = []
    for i in range(bsz):
        state_p = state_pred[i][state_pred_mask[i]][None]  # (1, nobj, 3)
        state_r = state_real[i][state_real_mask[i]][None]  # (1, nobj, 3)
        chamfer_dist = chamfer(state_p, state_r).item()
        chamfer_dist_list.append(chamfer_dist)
    chamfer_distance = np.array(chamfer_dist_list)  # (bsz,)
    return chamfer_distance

def box_loss(state, target):
    # state: (B, N, 3)
    # target: (2, 2)
    xmin, xmax, zmin, zmax = target[0, 0], target[0, 1], target[1, 0], target[1, 1]
    x_diff = torch.maximum(xmin - state[:, :, 0], torch.zeros_like(state[:, :, 0])) + \
        torch.maximum(state[:, :, 0] - xmax, torch.zeros_like(state[:, :, 0]))
    z_diff = torch.maximum(zmin - state[:, :, 2], torch.zeros_like(state[:, :, 2])) + \
        torch.maximum(state[:, :, 2] - zmax, torch.zeros_like(state[:, :, 2]))
    r_diff = (x_diff ** 2 + z_diff ** 2) ** 0.5  # (B, N)
    return r_diff.mean(dim=1)  # (B,)

def rope_penalty(state_pred, action, state_init, sim_real_ratio=10.0):
    bsz, n_look_forward, _ = action.shape
    x_start = action[:, :, 0]
    z_start = action[:, :, 1]
    action_point_2d = torch.stack([x_start, z_start], dim=-1)  # (bsz, n_look_forward, 2)
    state_2d = torch.cat([state_init[:, [0, 2]][None, None].repeat(bsz, 1, 1, 1),
                          state_pred[:, :-1, :, [0, 2]]], dim=1)  # (bsz, n_look_forward, max_nobj, 2)
    action_state_distance = torch.norm(action_point_2d[:, :, None] - state_2d, dim=-1).min(dim=-1).values  # (bsz, n_look_forward)
    pusher_size = 0.02 * sim_real_ratio
    action_state_distance = torch.maximum(action_state_distance - pusher_size, torch.zeros_like(action_state_distance))  # (bsz, n_look_forward)
    collision_penalty = torch.exp(-action_state_distance * 100.)  # (bsz, n_look_forward)
    return collision_penalty

def cloth_penalty(state_pred, action, state_init, sim_real_ratio=10.0):
    bsz, n_look_forward, _ = action.shape
    x_start = action[:, :, 0]
    z_start = action[:, :, 1]
    action_point_2d = torch.stack([x_start, z_start], dim=-1)  # (bsz, n_look_forward, 2)
    state_2d = state_init[:, [0, 2]]  # (bsz, n_look_forward, max_nobj, 2)
    action_state_distance = torch.norm(action_point_2d[:, :, None] - state_2d[None, None], dim=-1)
    action_state_min_dist = action_state_distance.min(dim=-1).values  # (bsz, n_look_forward)
    pusher_size = 0.005 * sim_real_ratio  # 5mm
    action_state_min_dist = torch.maximum(action_state_min_dist - pusher_size, torch.zeros_like(action_state_min_dist))  # (bsz, n_look_forward)
    action_state_max_dist = action_state_distance.max(dim=-1).values  # (bsz, n_look_forward)
    action_state_max_dist = torch.minimum(action_state_max_dist, torch.ones_like(action_state_max_dist) * 0.4 * sim_real_ratio)
    action_state_max_dist /= action_state_max_dist.max().item()
    collision_penalty = 1. - torch.exp(-action_state_min_dist * 100.) - action_state_max_dist * 0.2 # (bsz, n_look_forward)
    return collision_penalty

def granular_penalty(state_pred, action, state_init, sim_real_ratio=10.0):
    bsz, n_look_forward, _ = action.shape
    x_start = action[:, :, 0]  # (bsz, n_look_forward)
    z_start = action[:, :, 1]  # (bsz, n_look_forward)
    theta = action[:, :, 2]
    pusher_radius = 0.05 * sim_real_ratio
    delta_x = pusher_radius * torch.sin(theta)
    delta_z = -pusher_radius * torch.cos(theta)
    action_point_2d = torch.stack([
        x_start - delta_x, z_start - delta_z, 
        x_start - 0.75 * delta_x, z_start - 0.75 * delta_z,
        x_start - 0.5 * delta_x, z_start - 0.5 * delta_z,
        x_start - 0.25 * delta_x, z_start - 0.25 * delta_z,
        x_start, z_start,
        x_start + 0.25 * delta_x, z_start + 0.25 * delta_z,
        x_start + 0.5 * delta_x, z_start + 0.5 * delta_z,
        x_start + 0.75 * delta_x, z_start + 0.75 * delta_z,
        x_start + delta_x, z_start + delta_z], dim=-1)  # (bsz, n_look_forward, 9)
    state_2d = torch.cat([state_init[:, [0, 2]][None, None].repeat(bsz, 1, 1, 1),
                          state_pred[:, :-1, :, [0, 2]]], dim=1)  # (bsz, n_look_forward, max_nobj, 2)
    action_point_2d = action_point_2d.reshape(bsz, n_look_forward, 9, 2)  # (bsz, n_look_forward, 9, 2)
    action_state_distance = torch.norm(action_point_2d[:, :, :, None] - state_2d[:, :, None], dim=-1)  # (bsz, n_look_forward, 5, max_nobj)
    action_state_distance = action_state_distance.min(dim=-1).values.min(dim=-1).values  # (bsz, n_look_forward)
    pusher_size = 0.02 * sim_real_ratio
    action_state_distance = torch.maximum(action_state_distance - pusher_size, torch.zeros_like(action_state_distance))  # (bsz, n_look_forward)
    collision_penalty = torch.exp(-action_state_distance * 100.)  # (bsz, n_look_forward)
    return collision_penalty
