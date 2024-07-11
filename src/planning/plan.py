import argparse
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import time
import cv2
import math
import os
import yaml
import glob
import copy
from functools import partial

from planning.real_world.real_env import RealEnv
from planning.real_world.planner import Planner
from planning.forward_dynamics import dynamics
from planning.perception import PerceptionModule, get_state_cur
from planning.plan_utils import visualize_img, clip_actions, optimize_action_mppi, sample_action_seq
from planning.physics_param_optimizer import PhysicsParamOnlineOptimizer
from planning.losses import chamfer, box_loss, rope_penalty, cloth_penalty, granular_penalty

from dynamics.gnn.model import DynamicsPredictor
from dynamics.utils import set_seed


def running_cost(state, action, state_cur, error_func, penalty_func, bbox, **kwargs):  # tabletop coordinates
    # state: (bsz, n_look_forward, max_nobj, 3)
    # action: (bsz, n_look_forward, action_dim)
    # target_state: numpy.ndarray (n_target, 3)
    # state_cur: (max_nobj, 3)
    bsz = state.shape[0]
    n_look_forward = state.shape[1]

    state_flat = state.reshape(bsz * n_look_forward, state.shape[2], state.shape[3])
    error = error_func(state_flat).reshape(bsz, n_look_forward)
    error_weight = 2. / (error.max().item() + 1e-6)

    collision_penalty = penalty_func(state, action, state_cur)

    xmax = state.max(dim=2).values[:, :, 0]  # (bsz, n_look_forward)
    xmin = state.min(dim=2).values[:, :, 0]  # (bsz, n_look_forward)
    zmax = state.max(dim=2).values[:, :, 2]  # (bsz, n_look_forward)
    zmin = state.min(dim=2).values[:, :, 2]  # (bsz, n_look_forward)
    box_penalty = torch.stack([
        torch.maximum(xmin - bbox[0, 0], torch.zeros_like(xmin)),
        torch.maximum(bbox[0, 1] - xmax, torch.zeros_like(xmax)),
        torch.maximum(zmin - bbox[1, 0], torch.zeros_like(zmin)),
        torch.maximum(bbox[1, 1] - zmax, torch.zeros_like(zmax)),
    ], dim=-1)  # (bsz, n_look_forward, 4)
    box_penalty = torch.exp(-box_penalty * 100.).max(dim=-1).values  # (bsz, n_look_forward)

    reward = -error_weight * error[:, -1] - 5. * collision_penalty.mean(dim=1) - 5. * box_penalty.mean(dim=1)  # (bsz,)

    print(f'min error {error[:, -1].min().item()}, max reward {reward.max().item()}')
    out = {
        "reward_seqs": reward,
    }
    return out


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--task_config', type=str)
    arg_parser.add_argument('--resume', action='store_true')
    arg_parser.add_argument('--seed', type=int, default=43)
    arg_parser.add_argument('--use_ppo', action='store_true')
    args = arg_parser.parse_args()

    base_path = os.path.dirname(os.path.abspath(__file__))
    set_seed(args.seed)

    with open(args.task_config, 'r') as f:
        task_config = yaml.load(f, Loader=yaml.CLoader)['task_config']
    config_path = task_config['config']
    epoch = task_config['epoch']
    material = task_config['material']
    gripper_enable = task_config['gripper_enable']

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    train_config = config['train_config']
    dataset_config = config['dataset_config']
    model_config = config['model_config']
    material_config = config['material_config']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    exposure_time = 5
    env = RealEnv(
        task_config=task_config,
        WH=[1280, 720],
        capture_fps=5,
        obs_fps=5,
        n_obs_steps=1,
        use_robot=True,
        speed=50,
        gripper_enable=gripper_enable,
    )
    env.start(exposure_time=exposure_time)
    env.reset_robot()
    print('env started')
    time.sleep(exposure_time)
    print('start recording')
    env.calibrate(re_calibrate=False)  # just load the calibration file

    pm = PerceptionModule(task_config, device)

    action_lower_lim = torch.tensor(task_config['action_lower_lim'], dtype=torch.float32, device=device)
    action_upper_lim = torch.tensor(task_config['action_upper_lim'], dtype=torch.float32, device=device)

    run_name = dataset_config['data_name']
    save_dir = os.path.join(base_path, f"dump/vis/planning-{run_name}-model_{epoch}")
    if not args.resume and os.path.exists(save_dir) and len(glob.glob(os.path.join(save_dir, '*.npz'))) > 0:
        print('save dir already exists')
        env.stop()
        print('env stopped')
        return
    os.makedirs(save_dir, exist_ok=True)
    if args.resume:
        print('resume')
        n_resume = len(glob.glob(os.path.join(save_dir, 'interaction_*.npz')))
    else:
        n_resume = 0
    print('starting from iteration {}'.format(n_resume))
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(train_config['out_dir'], dataset_config['data_name'] ,'checkpoints', 'model_{}.pth'.format(epoch))

    model = DynamicsPredictor(model_config, material_config, dataset_config, device)
    model.to(device)

    model.eval()
    model.load_state_dict(torch.load(checkpoint_dir, map_location='cpu'))

    push_length = task_config['push_length']
    sim_real_ratio = task_config['sim_real_ratio']
    
    # target
    if task_config['target_type'] == 'pcd':
        pcd = o3d.io.read_point_cloud(os.path.join(base_path, task_config['target']))
        target_state = np.array(pcd.points) * sim_real_ratio
        target_state = target_state[:, [0, 2, 1]].copy()  # (x, y, z) -> (x, z, y)
        target_state[:, 1] *= -1  # (x, z, y) -> (x, -z, y)
        target_state = torch.tensor(target_state, dtype=torch.float32, device=device)
        target_box = None
        criteria = partial(chamfer, y=target_state[None])
    elif task_config['target_type'] == 'box':
        target_box = np.array([
            [float(task_config['target'][0]), float(task_config['target'][1])],
            [float(task_config['target'][2]), float(task_config['target'][3])]
        ])  # (x_min, x_max), (z_min, z_max)
        target_box = target_box * sim_real_ratio
        target_box = torch.tensor(target_box, dtype=torch.float32, device=device)
        target_state = None
        criteria = partial(box_loss, target=target_box)
    else:
        raise NotImplementedError(f"target type {task_config['target_type']} not implemented")
    
    # penalty
    if task_config['penalty_type'] == 'rope':
        penalty = partial(rope_penalty, sim_real_ratio=sim_real_ratio)
    elif task_config['penalty_type'] == 'cloth':
        penalty = partial(cloth_penalty, sim_real_ratio=sim_real_ratio)
    elif task_config['penalty_type'] == 'granular':
        penalty = partial(granular_penalty, sim_real_ratio=sim_real_ratio)
    else:
        raise NotImplementedError(f"penalty type {task_config['penalty_type']} not implemented")

    # bounding box penalty
    bbox_2d = np.array([
        [float(task_config['bbox'][0]), float(task_config['bbox'][1])],
        [float(task_config['bbox'][2]), float(task_config['bbox'][3])]
    ])  # (x_min, x_max), (z_min, z_max)
    bbox_2d = bbox_2d * sim_real_ratio
    running_cost_func = partial(running_cost, error_func=criteria, penalty_func=penalty, bbox=bbox_2d)

    n_actions = task_config['n_actions']  # total horizon size
    n_look_ahead = task_config['n_look_ahead']  # sliding window size
    n_sample = task_config['n_sample']
    n_sample_chunk = task_config['n_sample_chunk']

    n_chunk = np.ceil(n_sample / n_sample_chunk).astype(int)

    ppm_optimizer = PhysicsParamOnlineOptimizer(task_config, model, material, device, save_dir)

    noise_level = task_config['noise_level']
    reward_weight = task_config['reward_weight']
    planner_config = {
        'action_dim': len(action_lower_lim),
        'model_rollout_fn': partial(dynamics, model=model, device=device, ppm_optimizer=ppm_optimizer),
        'evaluate_traj_fn': running_cost_func,
        'sampling_action_seq_fn': partial(sample_action_seq, action_lower_lim=action_lower_lim, action_upper_lim=action_upper_lim, 
                                        n_sample=min(n_sample, n_sample_chunk), device=device, noise_level=noise_level, push_length=push_length),
        'clip_action_seq_fn': partial(clip_actions, action_lower_lim=action_lower_lim, action_upper_lim=action_upper_lim),
        'optimize_action_mppi_fn': partial(optimize_action_mppi, reward_weight=reward_weight, action_lower_lim=action_lower_lim, 
                                        action_upper_lim=action_upper_lim, push_length=push_length),
        'n_sample': min(n_sample, n_sample_chunk),
        'n_look_ahead': n_look_ahead,
        'n_update_iter': 1,
        'reward_weight': reward_weight,
        'action_lower_lim': action_lower_lim,
        'action_upper_lim': action_upper_lim,
        'planner_type': 'MPPI',
        'device': device,
        'verbose': False,
        'noise_level': noise_level,
        'rollout_best': True,
    }
    planner = Planner(planner_config)
    planner.total_chunks = n_chunk

    act_seq = torch.rand((planner_config['n_look_ahead'], action_upper_lim.shape[0]), device=device) * \
                (action_upper_lim - action_lower_lim) + action_lower_lim

    res_act_seq = torch.zeros((n_actions, action_upper_lim.shape[0]), device=device)

    if n_resume > 0:
        interaction_list = sorted(glob.glob(os.path.join(save_dir, 'interaction_*.npz')))
        for i in range(n_resume):
            res = np.load(interaction_list[i])
            act_save = res['act']
            state_init_save = res['state_init']
            state_pred_save = res['state_pred']
            state_real_save = res['state_real']
            res_act_seq[i] = torch.tensor(act_save, dtype=torch.float32, device=device)

    error_seq = []
    n_steps = n_actions
    for i in range(n_resume, n_actions):
        time1 = time.time()
        # get state
        state_cur, obj_pcd, rgb_vis, intr, extr = get_state_cur(env, pm, device, fps_radius=ppm_optimizer.fps_radius)

        if i == 0:
            error = criteria(torch.from_numpy(obj_pcd)[None].to(device)).item()
            print('error', error)
            error_seq.append(error)

        # get action
        res_all = []
        for ci in range(n_chunk):
            planner.chunk_id = ci
            res = planner.trajectory_optimization(state_cur, act_seq)
            for k, v in res.items():
                res[k] = v.detach().clone() if isinstance(v, torch.Tensor) else v
            res_all.append(res)
        res = planner.merge_res(res_all)

        # vis
        visualize_img(state_cur, res, rgb_vis, obj_pcd, material, intr, extr, 
                target_state=target_state, target_box=target_box,
                save_dir=save_dir, postfix=f'{i}_0', 
                task_config=task_config)

        # step state
        if gripper_enable:
            env.step_gripper(res['act_seq'][0].detach().cpu().numpy())
        else:
            env.step(res['act_seq'][0].detach().cpu().numpy())
        
        # update action
        res_act_seq[i] = res['act_seq'][0].detach().clone()
        act_seq = torch.cat(
            [
                res['act_seq'][1:],
                torch.rand((1, action_upper_lim.shape[0]), device=device) * (action_upper_lim - action_lower_lim) + action_lower_lim
            ], 
            dim=0
        )
        n_look_ahead = min(n_actions - i, planner_config['n_look_ahead'])
        act_seq = act_seq[:n_look_ahead]  # sliding window
        planner.n_look_ahead = n_look_ahead

        # save
        save = True
        if save:
            act_save = res['act_seq'][0].detach().cpu().numpy()  # (action_dim,)
            act_all_save = res['act_seq'].detach().cpu().numpy()  # (n_look_forward, action_dim)
            state_init_save = state_cur.detach().cpu().numpy()  # (max_nobj, 3)
            state_pred_save = res['best_model_output']['state_seqs'][0, 0].detach().cpu().numpy()  # (max_nobj, 3)
            state_pred_all_save = res['best_model_output']['state_seqs'][0].detach().cpu().numpy()  # (n_look_forward, max_nobj, 3)
            state_real, pcd_real, rgb_vis, _, _ = get_state_cur(env, pm, device, fps_radius=ppm_optimizer.fps_radius)
            state_real_save = state_real.detach().cpu().numpy()
            assert act_all_save.shape[0] == n_look_ahead
            assert state_pred_all_save.shape[0] == n_look_ahead
            np.savez(
                os.path.join(save_dir, f'interaction_{i}.npz'),
                act=act_save,
                act_all=act_all_save,
                state_pred=state_pred_save,
                state_pred_all=state_pred_all_save,
                pcd_real=pcd_real,
                state_real=state_real_save,
                state_init=state_init_save,
            )

            error = criteria(torch.from_numpy(pcd_real)[None].to(device)).item()
            print('error', error)
            error_seq.append(error)
            
            # vis
            visualize_img(state_cur, res, rgb_vis, obj_pcd, material, intr, extr, 
                    target_state=target_state, target_box=target_box,
                    state_after=state_real, save_dir=save_dir, postfix=f'{i}_1', 
                    task_config=task_config)

            # optimize physics parameter
            if args.use_ppo:
                ppm_optimizer.optimize(i, iterations=50)
            else:
                print("finished this step.")
                # time.sleep(10)
            
            time2 = time.time()
            print(f"step {i} time {time2 - time1}")

    print(f"final action sequence {res_act_seq}")
    print(f"final error sequence {error_seq}")

    with open(os.path.join(save_dir, 'stats.txt'), 'w') as f:
        f.write(f"final action sequence {res_act_seq}\n")
        f.write(f"final error sequence {error_seq}\n")

    env.stop()
    print('env stopped')

    # make video with cv2
    result = cv2.VideoWriter(
        os.path.join(save_dir, 'rgb_vis.mp4'), 
        cv2.VideoWriter_fourcc(*'mp4v'), 1, (1280, 720))

    for i in range(n_steps):
        rgb_vis = cv2.imread(os.path.join(save_dir, f'rgb_vis_{i}_0.png'))
        result.write(rgb_vis)
        rgb_vis = cv2.imread(os.path.join(save_dir, f'rgb_vis_{i}_1.png'))
        result.write(rgb_vis)

    result.release()
    print('video saved')

if __name__ == '__main__':
    with torch.no_grad():
        main()
