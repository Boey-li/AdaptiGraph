import torch
import time
import numpy as np

import sys
sys.path.append('.')
from dynamics.dataset.graph import construct_edges_from_states_batch
from dynamics.utils import pad_torch, truncate_graph
from planning.plan_utils import decode_action

@torch.no_grad()
def dynamics(state, action, model, device, ppm_optimizer, physics_param=None):
    time0 = time.time()
    max_n = ppm_optimizer.task_config['max_n']
    max_nR = ppm_optimizer.task_config['max_nR']
    n_his = ppm_optimizer.task_config['n_his']
    sim_real_ratio = ppm_optimizer.task_config['sim_real_ratio']
    push_length = ppm_optimizer.task_config['push_length']

    bsz = action.shape[0]
    n_look_forward = action.shape[1]

    decoded_action, action_repeat = decode_action(action, push_length=push_length)

    obj_kp = state[None, None].repeat(bsz, n_his, 1, 1)
    obj_kp_num = obj_kp.shape[2]
    eef_kp_num = ppm_optimizer.eef_num
    max_nobj = obj_kp_num
    max_neef = eef_kp_num
    material = ppm_optimizer.material

    pred_state_seq = torch.zeros((bsz, n_look_forward, max_nobj, 3), device=device)

    for li in range(n_look_forward):
        print(f"look forward iter {li}")
        
        if li > 0:
            obj_kp = pred_state_seq[:, li-1:li].detach().clone().repeat(1, n_his, 1, 1)

        y = (obj_kp[:, -1, :, 1]).min(dim=1).values  # (bsz,)

        if len(ppm_optimizer.task_config['pusher_points']) == 1:  # single point pusher
            eef_kp = torch.zeros((bsz, 1, 3))
            eef_kp[:, 0, 0] = decoded_action[:, li, 0]
            eef_kp[:, 0, 1] = y
            eef_kp[:, 0, 2] = decoded_action[:, li, 1]
            eef_kp_delta = torch.zeros((bsz, 1, 3))
            eef_kp_delta[:, 0, 0] = decoded_action[:, li, 2] - decoded_action[:, li, 0]
            eef_kp_delta[:, 0, 1] = 0
            eef_kp_delta[:, 0, 2] = decoded_action[:, li, 3] - decoded_action[:, li, 1]
        
        elif len(ppm_optimizer.task_config['pusher_points']) == 5:  # 5 point pusher
            eef_kp = torch.zeros((bsz, 5, 3))
            eef_kp[:, :, 1] = y[:, None]
            eef_kp_delta = torch.zeros((bsz, 5, 3))
            eef_kp_delta[:, :, 0] = (decoded_action[:, li, 2] - decoded_action[:, li, 0]).unsqueeze(1)
            eef_kp_delta[:, :, 1] = 0
            eef_kp_delta[:, :, 2] = (decoded_action[:, li, 3] - decoded_action[:, li, 1]).unsqueeze(1)

            x_start = decoded_action[:, li, 0]
            z_start = decoded_action[:, li, 1]
            theta = action[:, li, 2]

            pusher_points = ppm_optimizer.task_config['pusher_points']
            eef_kp[:, 0, 0] = x_start
            eef_kp[:, 1, 0] = x_start + float(pusher_points[1][1]) * sim_real_ratio * torch.sin(theta)
            eef_kp[:, 2, 0] = x_start + float(pusher_points[2][1]) * sim_real_ratio * torch.sin(theta)
            eef_kp[:, 3, 0] = x_start + float(pusher_points[3][1]) * sim_real_ratio * torch.sin(theta)
            eef_kp[:, 4, 0] = x_start + float(pusher_points[4][1]) * sim_real_ratio * torch.sin(theta)

            eef_kp[:, 0, 2] = z_start
            eef_kp[:, 1, 2] = z_start - float(pusher_points[1][1]) * sim_real_ratio * torch.cos(theta)
            eef_kp[:, 2, 2] = z_start - float(pusher_points[2][1]) * sim_real_ratio * torch.cos(theta)
            eef_kp[:, 3, 2] = z_start - float(pusher_points[3][1]) * sim_real_ratio * torch.cos(theta)
            eef_kp[:, 4, 2] = z_start - float(pusher_points[4][1]) * sim_real_ratio * torch.cos(theta)
        
        else:
            raise NotImplementedError("pusher not implemented")
        
        if ppm_optimizer.task_config['gripper_enable']:
            eef_kp[:, :, 1] += 0.01 * sim_real_ratio  # raise gripper by 1cm

        states = torch.zeros((bsz, n_his, max_nobj + max_neef, 3), device=device)
        states[:, :, :obj_kp_num] = obj_kp
        states[:, :, max_nobj : max_nobj + eef_kp_num] = eef_kp[:, None]

        states_delta = torch.zeros((bsz, max_nobj + max_neef, 3), device=device)
        states_delta[:, max_nobj : max_nobj + eef_kp_num] = eef_kp_delta

        attr_dim = 2
        attrs = torch.zeros((bsz, max_nobj + max_neef, attr_dim), dtype=torch.float32, device=device)
        attrs[:, :obj_kp_num, 0] = 1.
        attrs[:, max_nobj : max_nobj + eef_kp_num, 1] = 1.

        p_rigid = torch.zeros((bsz, max_n), dtype=torch.float32, device=device)

        p_instance = torch.zeros((bsz, max_nobj, max_n), dtype=torch.float32, device=device)
        instance_num = 1
        instance_kp_nums = [obj_kp_num]
        for i in range(bsz):
            ptcl_cnt = 0
            j_perm = np.random.permutation(instance_num)
            for j in range(instance_num):
                p_instance[i, ptcl_cnt:ptcl_cnt + instance_kp_nums[j], j_perm[j]] = 1
                ptcl_cnt += instance_kp_nums[j]

        state_mask = torch.zeros((bsz, max_nobj + max_neef), dtype=bool, device=device)
        state_mask[:, max_nobj : max_nobj + eef_kp_num] = True
        state_mask[:, :obj_kp_num] = True

        eef_mask = torch.zeros((bsz, max_nobj + max_neef), dtype=bool, device=device)
        eef_mask[:, max_nobj : max_nobj + eef_kp_num] = True

        obj_mask = torch.zeros((bsz, max_nobj,), dtype=bool, device=device)
        obj_mask[:, :obj_kp_num] = True

        material_dims = ppm_optimizer.material_dims
        material_indices = ppm_optimizer.material_indices
        physics_param = ppm_optimizer.physics_param if physics_param is None else physics_param
        adj_thresh = ppm_optimizer.adj_thresh

        material_idx = torch.zeros((bsz, max_nobj, len(material_indices)), dtype=int, device=device)
        material_idx[:, :obj_kp_num, material_indices[material]] = 1

        Rr, Rs = construct_edges_from_states_batch(states[:, -1], adj_thresh, mask=state_mask, tool_mask=eef_mask,
                topk=ppm_optimizer.task_config['topk'], connect_tools_all=ppm_optimizer.task_config['connect_tools_all'])
        Rr = pad_torch(Rr, max_nR, dim=1)
        Rs = pad_torch(Rs, max_nR, dim=1)

        graph = {
            # input information
            "state": states,  # (n_his, N+M, state_dim)
            "action": states_delta,  # (N+M, state_dim)

            # attr information
            "attrs": attrs,  # (N+M, attr_dim)
            "p_rigid": p_rigid,  # (n_instance,)
            "p_instance": p_instance,  # (N, n_instance)
            "obj_mask": obj_mask,  # (N,)
            "state_mask": state_mask,  # (N+M,)
            "eef_mask": eef_mask,  # (N+M,)

            "Rr": Rr,  # (bsz, max_nR, N)
            "Rs": Rs,  # (bsz, max_nR, N)

            "material_index": material_idx,  # (N, num_materials)
        }

        for material_name, material_dim in material_dims.items():
            if material_name in physics_param.keys():
                graph[material_name + '_physics_param'] = physics_param[material_name][None].repeat(bsz, 1)
            else:
                graph[material_name + '_physics_param'] = torch.zeros((bsz, material_dim), dtype=torch.float32)

        # rollout
        for ai in range(1, 1 + action_repeat[:, li].max().item()):
            graph = truncate_graph(graph)
            pred_state, pred_motion = model(**graph)

            repeat_mask = (action_repeat[:, li] == ai)
            pred_state_seq[repeat_mask, li] = pred_state[repeat_mask, :, :].clone()

            y_cur = pred_state[:, :, 1].min(dim=1).values
            eef_kp_cur = graph['state'][:, -1, max_nobj : max_nobj + eef_kp_num] + graph['action'][:, max_nobj : max_nobj + eef_kp_num]

            eef_kp_cur[:, :, 1] = y_cur[:, None]
            if ppm_optimizer.task_config['gripper_enable']:
                eef_kp_cur[:, :, 1] += 0.01 * sim_real_ratio  # raise gripper by 1cm

            states_cur = torch.cat([pred_state, eef_kp_cur], dim=1)
            Rr, Rs = construct_edges_from_states_batch(states_cur, adj_thresh, mask=graph['state_mask'], tool_mask=graph['eef_mask'], 
                    topk=ppm_optimizer.task_config['topk'], connect_tools_all=ppm_optimizer.task_config['connect_tools_all'])
            Rr = pad_torch(Rr, max_nR, dim=1)
            Rs = pad_torch(Rs, max_nR, dim=1)

            state_history = torch.cat([graph['state'][:, 1:], states_cur[:, None]], dim=1)

            new_graph = {
                "state": state_history,  # (bsz, n_his, N+M, state_dim)
                "action": graph["action"],  # (bsz, N+M, state_dim)
                
                "Rr": Rr,  # (bsz, n_rel, N+M)
                "Rs": Rs,  # (bsz, n_rel, N+M)
                
                "attrs": graph["attrs"],  # (bsz, N+M, attr_dim)
                "p_rigid": graph["p_rigid"],  # (bsz, n_instance)
                "p_instance": graph["p_instance"],  # (bsz, N, n_instance)
                "obj_mask": graph["obj_mask"],  # (bsz, N)
                "eef_mask": graph["eef_mask"],  # (bsz, N+M)
                "state_mask": graph["state_mask"],  # (bsz, N+M)
                "material_index": graph["material_index"],  # (bsz, N, num_materials)
            }
            for name in graph.keys():
                if name.endswith('_physics_param'):
                    new_graph[name] = graph[name]

            graph = new_graph

    out = {
        "state_seqs": pred_state_seq,  # (bsz, n_look_forward, max_nobj, 3)
        "action_seqs": decoded_action,  # (bsz, n_look_forward, action_dim)
    }
    time1 = time.time()
    print(f"dynamics time {time1 - time0}")
    return out


@torch.no_grad()
def dynamics_masked(state_init, state_mask, action, model, device, ppm_optimizer, physics_param=None):
    max_n = ppm_optimizer.task_config['max_n']
    max_nR = ppm_optimizer.task_config['max_nR']
    n_his = ppm_optimizer.task_config['n_his']
    sim_real_ratio = ppm_optimizer.task_config['sim_real_ratio']
    push_length = ppm_optimizer.task_config['push_length']

    bsz = state_init.shape[0]

    actions = action[:, None].to(device)  # (bsz, 1, action_dim)
    decoded_action, action_repeat = decode_action(actions, push_length=push_length)

    actions = actions.squeeze(1)  # (bsz, action_dim)
    decoded_action = decoded_action.squeeze(1)  # (bsz, action_dim)
    action_repeat = action_repeat.squeeze(1)  # (bsz,)

    state = state_init.clone().to(device)  # (bsz, nobj, 3)

    obj_kp = state[:, None].repeat(1, n_his, 1, 1)
    eef_kp_num = ppm_optimizer.eef_num
    max_nobj = state.shape[1]
    max_neef = eef_kp_num
    material = ppm_optimizer.material

    pred_state_seq = torch.zeros((bsz, max_nobj, 3), device=device)

    y = (state[:, :, 1] * state_mask).sum(dim=1) / state_mask.sum(dim=1)

    if len(ppm_optimizer.task_config['pusher_points']) == 1:  # single point pusher
        eef_kp = torch.zeros((bsz, 1, 3))
        eef_kp[:, 0, 0] = decoded_action[:, 0]
        eef_kp[:, 0, 1] = y
        eef_kp[:, 0, 2] = decoded_action[:, 1]
        eef_kp_delta = torch.zeros((bsz, 1, 3))
        eef_kp_delta[:, 0, 0] = decoded_action[:, 2] - decoded_action[:, 0]
        eef_kp_delta[:, 0, 1] = 0
        eef_kp_delta[:, 0, 2] = decoded_action[:, 3] - decoded_action[:, 1]
    
    elif len(ppm_optimizer.task_config['pusher_points']) == 5:  # 5 point pusher
        eef_kp = torch.zeros((bsz, 5, 3))
        eef_kp[:, :, 1] = y[:, None]
        eef_kp_delta = torch.zeros((bsz, 5, 3))
        eef_kp_delta[:, :, 0] = (decoded_action[:, 2] - decoded_action[:, 0]).unsqueeze(1)
        eef_kp_delta[:, :, 1] = 0
        eef_kp_delta[:, :, 2] = (decoded_action[:, 3] - decoded_action[:, 1]).unsqueeze(1)

        x_start = decoded_action[:, 0]
        z_start = decoded_action[:, 1]
        theta = action[:, 2]

        pusher_points = ppm_optimizer.task_config['pusher_points']
        eef_kp[:, 0, 0] = x_start
        eef_kp[:, 1, 0] = x_start + float(pusher_points[1][1]) * sim_real_ratio * torch.sin(theta)
        eef_kp[:, 2, 0] = x_start + float(pusher_points[2][1]) * sim_real_ratio * torch.sin(theta)
        eef_kp[:, 3, 0] = x_start + float(pusher_points[3][1]) * sim_real_ratio * torch.sin(theta)
        eef_kp[:, 4, 0] = x_start + float(pusher_points[4][1]) * sim_real_ratio * torch.sin(theta)

        eef_kp[:, 0, 2] = z_start
        eef_kp[:, 1, 2] = z_start - float(pusher_points[1][1]) * sim_real_ratio * torch.cos(theta)
        eef_kp[:, 2, 2] = z_start - float(pusher_points[2][1]) * sim_real_ratio * torch.cos(theta)
        eef_kp[:, 3, 2] = z_start - float(pusher_points[3][1]) * sim_real_ratio * torch.cos(theta)
        eef_kp[:, 4, 2] = z_start - float(pusher_points[4][1]) * sim_real_ratio * torch.cos(theta)
    
    else:
        raise NotImplementedError("pusher not implemented")
    
    if ppm_optimizer.task_config['gripper_enable']:
        eef_kp[:, :, 1] += 0.01 * sim_real_ratio  # raise gripper by 1cm

    states = torch.zeros((bsz, n_his, max_nobj + max_neef, 3), device=device)
    states[:, :, :max_nobj] = obj_kp
    states[:, :, max_nobj : max_nobj + eef_kp_num] = eef_kp[:, None]

    states_delta = torch.zeros((bsz, max_nobj + max_neef, 3), device=device)
    states_delta[:, max_nobj : max_nobj + eef_kp_num] = eef_kp_delta

    attr_dim = 2
    attrs = torch.zeros((bsz, max_nobj + max_neef, attr_dim), dtype=torch.float32, device=device)
    attrs[:, :max_nobj, 0][state_mask] = 1.
    attrs[:, max_nobj : max_nobj + eef_kp_num, 1] = 1.

    p_rigid = torch.zeros((bsz, max_n), dtype=torch.float32, device=device)

    p_instance = torch.zeros((bsz, max_nobj, max_n), dtype=torch.float32, device=device)
    instance_num = 1
    for i in range(bsz):
        instance_kp_nums = [state_mask[i].sum().item()]
        ptcl_cnt = 0
        j_perm = np.random.permutation(instance_num)
        for j in range(instance_num):
            p_instance[i, ptcl_cnt:ptcl_cnt + instance_kp_nums[j], j_perm[j]] = 1
            ptcl_cnt += instance_kp_nums[j]

    state_mask_new = torch.zeros((bsz, max_nobj + max_neef), dtype=bool, device=device)
    state_mask_new[:, max_nobj : max_nobj + eef_kp_num] = True
    state_mask_new[:, :max_nobj] = state_mask.clone()

    eef_mask = torch.zeros((bsz, max_nobj + max_neef), dtype=bool, device=device)
    eef_mask[:, max_nobj : max_nobj + eef_kp_num] = True

    obj_mask = state_mask.clone()
    
    material_dims = ppm_optimizer.material_dims
    material_indices = ppm_optimizer.material_indices
    physics_param = ppm_optimizer.physics_param if physics_param is None else physics_param
    adj_thresh = ppm_optimizer.adj_thresh

    material_idx = torch.zeros((bsz, max_nobj, len(material_indices)), dtype=int, device=device)
    material_idx[:, :max_nobj, material_indices[material]][state_mask] = 1

    Rr, Rs = construct_edges_from_states_batch(states[:, -1], adj_thresh, mask=state_mask_new, tool_mask=eef_mask, 
            topk=ppm_optimizer.task_config['topk'], connect_tools_all=ppm_optimizer.task_config['connect_tools_all'])
    Rr = pad_torch(Rr, max_nR, dim=1)
    Rs = pad_torch(Rs, max_nR, dim=1)

    graph = {
        # input information
        "state": states,  # (B, n_his, N+M, state_dim)
        "action": states_delta,  # (B, N+M, state_dim)

        # attr information
        "attrs": attrs,  # (B, N+M, attr_dim)
        "p_rigid": p_rigid,  # (B, n_instance,)
        "p_instance": p_instance,  # (B, N, n_instance)
        "obj_mask": obj_mask,  # (B, N,)
        "state_mask": state_mask_new,  # (B, N+M,)
        "eef_mask": eef_mask,  # (B, N+M,)

        "Rr": Rr,  # (bsz, max_nR, N)
        "Rs": Rs,  # (bsz, max_nR, N)

        "material_index": material_idx,  # (N, num_materials)
    }

    for material_name, material_dim in material_dims.items():
        if material_name in physics_param.keys():
            graph[material_name + '_physics_param'] = physics_param[material_name][None].repeat(bsz, 1)
        else:
            import ipdb; ipdb.set_trace()
            # graph[material_name + '_physics_param'] = torch.zeros((bsz, material_dim), dtype=torch.float32)

    # rollout
    for ai in range(1, 1 + action_repeat.max().item()):
        # print(f"rollout iter {i}")
        graph = truncate_graph(graph)
        pred_state, pred_motion = model(**graph)

        repeat_mask = (action_repeat == ai)
        pred_state_seq[repeat_mask] = pred_state[repeat_mask, :, :].clone()

        y_cur = (pred_state[:, :, 1] * state_mask).sum(dim=1) / state_mask.sum(dim=1)
        eef_kp_cur = graph['state'][:, -1, max_nobj : max_nobj + eef_kp_num] + graph['action'][:, max_nobj : max_nobj + eef_kp_num]

        eef_kp_cur[:, :, 1] = y_cur[:, None]
        if ppm_optimizer.task_config['gripper_enable']:
            eef_kp_cur[:, :, 1] += 0.01 * sim_real_ratio  # raise gripper by 1cm

        states_cur = torch.cat([pred_state, eef_kp_cur], dim=1)
        Rr, Rs = construct_edges_from_states_batch(states_cur, adj_thresh, mask=graph['state_mask'], tool_mask=graph['eef_mask'],\
                topk=ppm_optimizer.task_config['topk'], connect_tools_all=ppm_optimizer.task_config['connect_tools_all'])
        Rr = pad_torch(Rr, max_nR, dim=1)
        Rs = pad_torch(Rs, max_nR, dim=1)

        state_history = torch.cat([graph['state'][:, 1:], states_cur[:, None]], dim=1)

        new_graph = {
            "state": state_history,  # (bsz, n_his, N+M, state_dim)
            "action": graph["action"],  # (bsz, N+M, state_dim)
            
            "Rr": Rr,  # (bsz, n_rel, N+M)
            "Rs": Rs,  # (bsz, n_rel, N+M)
            
            "attrs": graph["attrs"],  # (bsz, N+M, attr_dim)
            "p_rigid": graph["p_rigid"],  # (bsz, n_instance)
            "p_instance": graph["p_instance"],  # (bsz, N, n_instance)
            "obj_mask": graph["obj_mask"],  # (bsz, N)
            "eef_mask": graph["eef_mask"],  # (bsz, N+M)
            "state_mask": graph["state_mask"],  # (bsz, N+M)
            "material_index": graph["material_index"],  # (bsz, N, num_materials)
        }
        for name in graph.keys():
            if name.endswith('_physics_param'):
                new_graph[name] = graph[name]

        graph = new_graph

    out = {
        "state_seqs": pred_state_seq,  # (bsz, max_nobj, 3)
        "action_seqs": decoded_action,  # (bsz, action_dim)
    }
    return out


