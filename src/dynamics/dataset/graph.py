import numpy as np
import copy
import torch

from dgl.geometry import farthest_point_sampler
from dynamics.utils import fps_rad_idx

def fps(obj_kp_start, max_nobj, fps_radius_range, verbose=False):
        ## farthest point sampling
        particle_tensor = torch.from_numpy(obj_kp_start).float().unsqueeze(0) # [1, N, 3]
        fps_idx_tensor = farthest_point_sampler(particle_tensor, min(max_nobj, particle_tensor.shape[1]),
                                                start_idx=np.random.randint(0, particle_tensor.shape[1]))[0]
        fps_idx_1 = fps_idx_tensor.numpy().astype(np.int32) # (max_nobj, )
        
        ## downsample to uniform radius
        downsample_particle = particle_tensor[0, fps_idx_1].numpy() # (max_nobj, 3)
        
        ## choose fps radius
        if type(fps_radius_range) == float:
            fps_radius = fps_radius_range
        elif len(fps_radius_range) == 2:
            fps_radius = np.random.uniform(fps_radius_range[0], fps_radius_range[1])
        else:
            raise ValueError(f"Invalid fps_radius_range: {fps_radius_range}.")
        
        _, fps_idx_2 = fps_rad_idx(downsample_particle, fps_radius)
        fps_idx_2 = fps_idx_2.astype(np.int32)
        fps_idx = fps_idx_1[fps_idx_2]
        
        if verbose:
            print(f"FPS num particles: {len(fps_idx)} with index list \n {fps_idx}. \n")
        
        # fps index list
        fps_idx_list = np.array(fps_idx)  # (n_fps, )
        
        return fps_idx_list

def construct_edges_from_states(states, adj_thresh, mask, tool_mask, topk=10, connect_tools_all=False):
    # :param states: (N, state_dim) torch tensor
    # :param adj_thresh: float
    # :param mask: (N) torch tensor, true when index is a valid particle
    # :param tool_mask: (N) torch tensor, true when index is a valid tool particle
    # :return:
    # - Rr: (n_rel, N) torch tensor
    # - Rs: (n_rel, N) torch tensor

    N, state_dim = states.shape
    s_receiv = states[:, None, :].repeat(1, N, 1)
    s_sender = states[None, :, :].repeat(N, 1, 1)

    # dis: particle_num x particle_num
    # adj_matrix: particle_num x particle_num
    threshold = adj_thresh * adj_thresh
    s_diff = s_receiv - s_sender  # (N, N, state_dim)
    dis = torch.sum(s_diff ** 2, -1)
    mask_1 = mask[:, None].repeat(1, N)
    mask_2 = mask[None, :].repeat(N, 1)
    mask_12 = mask_1 * mask_2
    dis[~mask_12] = 1e10  # avoid invalid particles to particles relations
    tool_mask_1 = tool_mask[:, None].repeat(1, N)
    tool_mask_2 = tool_mask[None, :].repeat(N, 1)
    tool_mask_12 = tool_mask_1 * tool_mask_2
    dis[tool_mask_12] = 1e10  # avoid tool to tool relations

    obj_tool_mask_1 = tool_mask_1 * mask_2  # particle sender, tool receiver
    obj_tool_mask_2 = tool_mask_2 * mask_1  # particle receiver, tool sender

    adj_matrix = ((dis - threshold) < 0).float()

    # add topk constraints
    topk = min(dis.shape[-1], topk)
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix

    if connect_tools_all:
        adj_matrix[obj_tool_mask_1] = 0
        adj_matrix[obj_tool_mask_2] = 1
        adj_matrix[tool_mask_12] = 0  # avoid tool to tool relations

    n_rels = adj_matrix.sum().long().item()
    rels_idx = torch.arange(n_rels).to(device=states.device, dtype=torch.long)
    rels = adj_matrix.nonzero()
    Rr = torch.zeros((n_rels, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((n_rels, N), device=states.device, dtype=states.dtype)
    Rr[rels_idx, rels[:, 0]] = 1
    Rs[rels_idx, rels[:, 1]] = 1
    return Rr, Rs

def construct_edges_from_states_batch(states, adj_thresh, mask, tool_mask, topk=10, connect_tools_all=False):  
    # :param states: (B, N, state_dim) torch tensor
    # :param adj_thresh: (B, ) torch tensor
    # :param mask: (B, N) torch tensor, true when index is a valid particle
    # :param tool_mask: (B, N) torch tensor, true when index is a valid tool particle
    # :return:
    # - Rr: (B, n_rel, N) torch tensor
    # - Rs: (B, n_rel, N) torch tensor

    B, N, state_dim = states.shape
    s_receiv = states[:, :, None, :].repeat(1, 1, N, 1)
    s_sender = states[:, None, :, :].repeat(1, N, 1, 1)

    # dis: B x particle_num x particle_num
    # adj_matrix: B x particle_num x particle_num
    if isinstance(adj_thresh, float):
        adj_thresh = torch.tensor(adj_thresh, device=states.device, dtype=states.dtype).repeat(B)
    threshold = adj_thresh * adj_thresh
    s_diff = s_receiv - s_sender  # (N, N, state_dim)
    dis = torch.sum(s_diff ** 2, -1)
    mask_1 = mask[:, :, None].repeat(1, 1, N)
    mask_2 = mask[:, None, :].repeat(1, N, 1)
    mask_12 = mask_1 * mask_2
    dis[~mask_12] = 1e10  # avoid invalid particles to particles relations
    tool_mask_1 = tool_mask[:, :, None].repeat(1, 1, N)
    tool_mask_2 = tool_mask[:, None, :].repeat(1, N, 1)
    tool_mask_12 = tool_mask_1 * tool_mask_2
    dis[tool_mask_12] = 1e10  # avoid tool to tool relations

    obj_tool_mask_1 = tool_mask_1 * mask_2  # particle sender, tool receiver
    obj_tool_mask_2 = tool_mask_2 * mask_1  # particle receiver, tool sender

    obj_pad_tool_mask_1 = tool_mask_1 * (~tool_mask_2)

    adj_matrix = ((dis - threshold[:, None, None]) < 0).float()

    # add topk constraints
    topk = min(dis.shape[-1], topk)
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix

    if connect_tools_all:
        batch_mask = (adj_matrix[obj_pad_tool_mask_1].reshape(B, -1).sum(-1) > 0)[:, None, None].repeat(1, N, N)
        batch_obj_tool_mask_1 = obj_tool_mask_1 * batch_mask  # (B, N, N)
        neg_batch_obj_tool_mask_1 = obj_tool_mask_1 * (~batch_mask)  # (B, N, N)
        batch_obj_tool_mask_2 = obj_tool_mask_2 * batch_mask  # (B, N, N)
        neg_batch_obj_tool_mask_2 = obj_tool_mask_2 * (~batch_mask)  # (B, N, N)

        adj_matrix[batch_obj_tool_mask_1] = 0
        adj_matrix[batch_obj_tool_mask_2] = 1
        adj_matrix[neg_batch_obj_tool_mask_1] = 0
        adj_matrix[neg_batch_obj_tool_mask_2] = 0

    n_rels = adj_matrix.sum(dim=(1,2))
    n_rel = n_rels.max().long().item()
    rels_idx = []
    rels_idx = [torch.arange(n_rels[i]) for i in range(B)]
    rels_idx = torch.hstack(rels_idx).to(device=states.device, dtype=torch.long)
    rels = adj_matrix.nonzero()
    Rr = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1
    Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1
    return Rr, Rs
        