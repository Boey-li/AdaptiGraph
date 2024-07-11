import os
import math
import torch
import numpy as np
import cv2
import torch.nn.functional as F

from dynamics.dataset.graph import construct_edges_from_states


def decode_action(action, push_length=0.10):
    x_start = action[:, :, 0]
    z_start = action[:, :, 1]
    theta = action[:, :, 2]
    length = action[:, :, 3].detach()
    action_repeat = length.to(torch.int32)
    x_end = x_start - push_length * torch.cos(theta)
    z_end = z_start - push_length * torch.sin(theta)
    decoded_action = torch.stack([x_start, z_start, x_end, z_end], dim=-1)
    return decoded_action, action_repeat

def decode_action_single(action, push_length=0.10):
    x_start = action[0]
    z_start = action[1]
    theta = action[2]
    action_repeat = int(action[3])
    x_end = x_start - push_length * action_repeat * np.cos(theta)
    z_end = z_start - push_length * action_repeat * np.sin(theta)
    return x_start, z_start, x_end, z_end

def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)


def clip_actions(action, action_lower_lim, action_upper_lim):
    action_new = action.clone()
    action_new[..., 2] = angle_normalize(action[..., 2])
    action_new.data.clamp_(action_lower_lim, action_upper_lim)
    return action_new


def sample_action_seq(act_seq, action_lower_lim, action_upper_lim, n_sample, device, iter_index=0, noise_level=0.3, push_length=0.10):
    if iter_index == 0:
        # resample completely
        act_seqs = torch.rand((n_sample, act_seq.shape[0], act_seq.shape[1]), device=device) * \
            (action_upper_lim - action_lower_lim) + action_lower_lim
    else:
        n_look_ahead = act_seq.shape[0]
        
        assert act_seq.shape[-1] == 4  # (x, y, theta, length)
        act_seqs = torch.stack([act_seq.clone()] * n_sample)
        xs = act_seqs[:, :, 0]
        ys = act_seqs[:, :, 1]
        thetas = act_seqs[:, :, 2]
        lengths = act_seqs[:, :, 3]
        
        x_ends = xs - lengths * push_length * torch.cos(thetas)
        y_ends = ys - lengths * push_length * torch.sin(thetas)

        for i in range(n_look_ahead):
            noise_sample = torch.normal(0, noise_level, (n_sample, 4), device=device)
            beta = 0.1 * (10 ** i)
            act_residual = beta * noise_sample
            
            xs_i = xs[:, i] + act_residual[:, 0]
            ys_i = ys[:, i] + act_residual[:, 1]
            x_ends_i = x_ends[:, i] + act_residual[:, 2]
            y_ends_i = y_ends[:, i] + act_residual[:, 3]

            thetas_i = torch.atan2(ys_i - y_ends_i, xs_i - x_ends_i)
            lengths_i = torch.norm(torch.stack([x_ends_i - xs_i, y_ends_i - ys_i], dim=-1), dim=-1).clone() / push_length

            act_seq_i = torch.stack([xs_i, ys_i, thetas_i, lengths_i], dim=-1)
            act_seq_i = clip_actions(act_seq_i, action_lower_lim, action_upper_lim)
            act_seqs[1:, i] = act_seq_i[1:].clone()

    return act_seqs  # (n_sample, n_look_ahead, action_dim)


def optimize_action_mppi(act_seqs, reward_seqs, reward_weight=100.0, action_lower_lim=None, action_upper_lim=None, push_length=0.10):
    weight_seqs = F.softmax(reward_seqs * reward_weight, dim=0).unsqueeze(-1)

    assert act_seqs.shape[-1] == 4  # (x, y, theta, length)
    xs = act_seqs[:, :, 0]
    ys = act_seqs[:, :, 1]
    thetas = act_seqs[:, :, 2]
    lengths = act_seqs[:, :, 3]
    x_ends = xs - lengths * push_length * torch.cos(thetas)
    y_ends = ys - lengths * push_length * torch.sin(thetas)

    x = torch.sum(weight_seqs * xs, dim=0)  # (n_look_ahead,)
    y = torch.sum(weight_seqs * ys, dim=0)  # (n_look_ahead,)
    x_end = torch.sum(weight_seqs * x_ends, dim=0)  # (n_look_ahead,)
    y_end = torch.sum(weight_seqs * y_ends, dim=0)  # (n_look_ahead,)

    theta = torch.atan2(y - y_end, x - x_end)  # (n_look_ahead,)
    length = torch.norm(torch.stack([x_end - x, y_end - y], dim=-1), dim=-1) / push_length  # (n_look_ahead,)

    act_seq = torch.stack([x, y, theta, length], dim=-1)  # (n_look_ahead, action_dim)
    act_seq = clip_actions(act_seq, action_lower_lim, action_upper_lim)
    return act_seq


def visualize_img(state_init, res, rgb_vis, obj_pcd, material, intr, extr, 
        target_state=None, target_box=None, state_after=None, 
        save_dir=None, postfix=None, 
        task_config=None):
    # state_init: (n_points, 3)
    # state: (n_look_forward, n_points, 3)
    # target_state: (n_points_raw, 3)
    # rgb_vis: np.ndarray (H, W, 3)
    # obj_pcd: np.ndarray (n_points, 3)
    push_length = task_config['push_length']
    sim_real_ratio = task_config['sim_real_ratio']
    adj_thresh = task_config['adj_thresh']
    topk = task_config['topk']
    connect_tools_all = task_config['connect_tools_all']

    def project(points, intr, extr):
        # extr: (4, 4)
        # intr: (3, 3)
        # points: (n_points, 3)

        # transform points back to table coordinates
        points = points.copy()
        points[:, 1] *= -1  # (x, -z, y) -> (x, z, y)
        points = points[:, [0, 2, 1]].copy()  # (x, z, y) -> (x, y, z)
        points = points / sim_real_ratio  # sim 2 real

        # project
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        points = points @ extr.T  # (n_points, 4)
        points = points[:, :3] / points[:, 2:3]  # (n_points, 3)
        points = points @ intr.T
        points = points[:, :2] / points[:, 2:3]  # (n_points, 2)
        return points

    # best result
    action_best = res['act_seq']  # (n_look_forward, action_dim)
    state_best = res['best_model_output']['state_seqs'][0]  # (n_look_forward, max_nobj, 3)

    # plot
    action, repeat = decode_action(action_best.unsqueeze(0), push_length=push_length)  # (1, n_look_forward, action_dim)
    action = action[0]  # (n_look_forward, action_dim)
    repeat = repeat[0, 0].item()

    state_init_vis = state_init.detach().cpu().numpy()  # (n_points, 3)
    state_vis = state_best[0].detach().cpu().numpy()  # (n_points, 3)
    if target_state is not None:
        target_state_vis = target_state.detach().cpu().numpy()  # (n_target_points, 3)
    action_vis = action[0].detach().cpu().numpy()  # (action_dim,)

    Rr, Rs = construct_edges_from_states(torch.from_numpy(state_init_vis), adj_thresh, 
                mask=torch.ones(state_init_vis.shape[0], dtype=bool),
                tool_mask=torch.zeros(state_init_vis.shape[0], dtype=bool),
                topk=topk, connect_tools_all=connect_tools_all)
    Rr = Rr.numpy()  # (n_rel, n_points)
    Rs = Rs.numpy()  # (n_rel, n_points)

    Rr_best, Rs_best = construct_edges_from_states(torch.from_numpy(state_vis), adj_thresh,
                mask=torch.ones(state_vis.shape[0], dtype=bool),
                tool_mask=torch.zeros(state_vis.shape[0], dtype=bool),
                topk=topk, connect_tools_all=connect_tools_all)
    Rr_best = Rr_best.numpy()  # (n_rel, n_points)
    Rs_best = Rs_best.numpy()  # (n_rel, n_points)

    if state_after is not None:
        state_after_vis = state_after.detach().cpu().numpy()  # (n_points, 3)

        # construct relations
        Rr_after, Rs_after = construct_edges_from_states(torch.from_numpy(state_after_vis), adj_thresh,
                    mask=torch.ones(state_after_vis.shape[0], dtype=bool),
                    tool_mask=torch.zeros(state_after_vis.shape[0], dtype=bool),
                    topk=topk, connect_tools_all=connect_tools_all)
        Rr_after = Rr_after.numpy()
        Rs_after = Rs_after.numpy()

        Rr = Rr_after.copy()
        Rs = Rs_after.copy()

        state_init_vis = state_after_vis.copy()

    # plot state_init_vis, Rr, Rs, action_vis, state_vis, target_state_vis on rgb_vis

    # preparation
    state_init_proj = project(state_init_vis, intr, extr)
    state_proj = project(state_vis, intr, extr)
    if target_state is not None:
        target_state_proj = project(target_state_vis, intr, extr)

    # visualize
    rgb_orig = rgb_vis.copy()

    color_start = (202, 63, 41)
    color_action = (27, 74, 242)
    color_pred = (237, 158, 49)
    color_target = (26, 130, 81)

    # starting state
    point_size = 5
    for k in range(state_init_proj.shape[0]):
        cv2.circle(rgb_vis, (int(state_init_proj[k, 0]), int(state_init_proj[k, 1])), point_size, 
            color_start, -1)

    # starting edges
    edge_size = 2
    for k in range(Rr.shape[0]):
        if Rr[k].sum() == 0: continue
        receiver = Rr[k].argmax()
        sender = Rs[k].argmax()
        cv2.line(rgb_vis, 
            (int(state_init_proj[receiver, 0]), int(state_init_proj[receiver, 1])), 
            (int(state_init_proj[sender, 0]), int(state_init_proj[sender, 1])), 
            color_start, edge_size)
    
    # action arrow
    x_start = action_vis[0]
    z_start = action_vis[1]
    x_end = action_vis[2]
    z_end = action_vis[3]
    x_delta = x_end - x_start
    z_delta = z_end - z_start
    y = state_init[:, 1].mean().item()
    arrow_size = 2
    tip_length = 0.5
    for i in range(repeat):
        action_start_point = np.array([x_start + i * x_delta, y, z_start + i * z_delta])
        action_end_point = np.array([x_end + i * x_delta, y, z_end + i * z_delta])
        action_start_point_proj = project(action_start_point[None], intr, extr)[0]
        action_end_point_proj = project(action_end_point[None], intr, extr)[0]
        cv2.arrowedLine(rgb_vis,
            (int(action_start_point_proj[0]), int(action_start_point_proj[1])),
            (int(action_end_point_proj[0]), int(action_end_point_proj[1])),
            color_action, arrow_size, tipLength=tip_length)

    rgb_overlay = rgb_vis.copy()

    # target point cloud
    if target_state is not None:
        for k in range(target_state_proj.shape[0]):
            cv2.circle(rgb_vis, (int(target_state_proj[k, 0]), int(target_state_proj[k, 1])), point_size, 
                color_target, -1)
    
    if target_box is not None:
        x_min, x_max, z_min, z_max = target_box[0, 0].item(), target_box[0, 1].item(), target_box[1, 0].item(), target_box[1, 1].item()
        edge = 0.03
        rect_1 = np.array([[x_min - edge, 0, z_min - edge], [x_min + edge, 0, z_min - edge], [x_min + edge, 0, z_max + edge], [x_min - edge, 0, z_max + edge]])
        rect_2 = np.array([[x_max - edge, 0, z_min - edge], [x_max + edge, 0, z_min - edge], [x_max + edge, 0, z_max + edge], [x_max - edge, 0, z_max + edge]])
        rect_3 = np.array([[x_min + edge, 0, z_min - edge], [x_max - edge, 0, z_min - edge], [x_max - edge, 0, z_min + edge], [x_min + edge, 0, z_min + edge]])
        rect_4 = np.array([[x_min + edge, 0, z_max - edge], [x_max - edge, 0, z_max - edge], [x_max - edge, 0, z_max + edge], [x_min + edge, 0, z_max + edge]])

        rect_1_proj = project(rect_1, intr, extr)
        rect_2_proj = project(rect_2, intr, extr)
        rect_3_proj = project(rect_3, intr, extr)
        rect_4_proj = project(rect_4, intr, extr)

        cv2.fillConvexPoly(rgb_vis, rect_1_proj.astype(np.int32), color_target)
        cv2.fillConvexPoly(rgb_vis, rect_2_proj.astype(np.int32), color_target)
        cv2.fillConvexPoly(rgb_vis, rect_3_proj.astype(np.int32), color_target)
        cv2.fillConvexPoly(rgb_vis, rect_4_proj.astype(np.int32), color_target)

    # predicted state
    for k in range(state_proj.shape[0]):
        cv2.circle(rgb_vis, (int(state_proj[k, 0]), int(state_proj[k, 1])), point_size, 
            color_pred, -1)
    
    # predicted edges
    for k in range(Rr_best.shape[0]):
        if Rr_best[k].sum() == 0: continue
        receiver = Rr_best[k].argmax()
        sender = Rs_best[k].argmax()
        cv2.line(rgb_vis, 
            (int(state_proj[receiver, 0]), int(state_proj[receiver, 1])), 
            (int(state_proj[sender, 0]), int(state_proj[sender, 1])), 
            color_pred, edge_size)
    
    rgb_vis = cv2.addWeighted(rgb_overlay, 0.5, rgb_vis, 0.5, 0)
    
    if save_dir is not None:
        cv2.imwrite(os.path.join(save_dir, f'rgb_vis_{postfix}.png'), rgb_vis)
        cv2.imwrite(os.path.join(save_dir, f'rgb_orig_{postfix}.png'), rgb_orig)
