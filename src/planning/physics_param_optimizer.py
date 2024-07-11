import torch
import numpy as np
import os
import glob
import copy
import cma
from functools import partial

from skopt import gp_minimize
from skopt.learning.gaussian_process.kernels import WhiteKernel, RBF, Matern
from skopt.learning import GaussianProcessRegressor
from skopt.utils import expected_minimum

from planning.forward_dynamics import dynamics_masked
from planning.losses import mean_chamfer


class PhysicsParamOnlineOptimizer:
    def __init__(self, task_config, model, material, device, save_dir):
        self.task_config = task_config
        self.model = model
        self.material = material
        self.device = device
        self.save_dir = save_dir

        self.physics_param = dict()
        self.material_indices = task_config['material_indices']
        self.material_dims = task_config['material_dims']
        self.fps_radius = task_config['fps_radius']
        self.adj_thresh = task_config['adj_thresh']
        self.eef_num = task_config['eef_num']
        self.physics_param[self.material] = torch.tensor([0.5], device=device).repeat(self.material_dims[self.material])
    
    def optimize(self, i, iterations=50):
        # read
        interaction_list = sorted(glob.glob(os.path.join(self.save_dir, 'interaction_*.npz')))
        assert len(interaction_list) == i + 1, f"interaction list {len(interaction_list)} != {i + 1}"

        act = []
        state_init = []
        state_pred = []
        state_real = []
        for ii in range(len(interaction_list)):
            res = np.load(interaction_list[ii])
            act_save = res['act']
            state_init_save = res['state_init']
            state_pred_save = res['state_pred']
            state_real_save = res['state_real']

            act.append(act_save)
            state_init.append(state_init_save)
            state_pred.append(state_pred_save)
            state_real.append(state_real_save)

        print('optimizing physics param...')
        optimize_func = optimize if self.material_dims[self.material] == 1 else optimize_cma
        assert iterations > 0  # assure optimize_func is not short-circuited
        ppm, error, error_init, res = optimize_func(
            self, act, state_init, state_pred, state_real, iter_idx=i, iterations=iterations, return_res=True)

        print('new physics param', ppm)
        self.physics_param[self.material] = torch.tensor(ppm, dtype=torch.float32, device=self.device)
        self.physics_param[self.material] = torch.clamp(self.physics_param[self.material], -0.2, 1.2)

        # save
        optim_save_dir = os.path.join(self.save_dir, f'ppo_{i}.npz')
        np.savez(
            optim_save_dir,
            physics_param=np.array(ppm),
            error=error,
            error_init=error_init,
        )


def optimize(ppm_optimizer, actions, state_init_list, state_pred_list, state_real_list, 
             num_optim_trials=1, iter_idx=0, iterations=50, return_res=False):
    if iterations < 0:
        iterations = 200  # the maximum number of iterations
    
    physics_param_init = dict()
    for material_name, material_dim in ppm_optimizer.material_dims.items():
        physics_param_init[material_name] = ppm_optimizer.physics_param[material_name]

    init_error = dynamics_error(physics_param_init, ppm_optimizer, state_init_list, state_real_list, actions)
    print(f"init error {init_error}")
    
    if iterations == 0:
        return init_error

    optimal_physics_param_list = []
    for _ in range(num_optim_trials):
        ## optimize
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=(0.2 * init_error) ** 2)
        base_estimator = GaussianProcessRegressor(kernel=kernel, normalize_y=True, noise="gaussian", n_restarts_optimizer=10)
        res = gp_minimize(partial(
                dynamics_error,
                ppm_optimizer=ppm_optimizer, state_init_list=state_init_list, state_real_list=state_real_list, actions=actions,
            ),
            [(-0.2, 1.2)] * sum([material_dim for material_dim in ppm_optimizer.material_dims.values()]),
            base_estimator=base_estimator,  # the model
            acq_func="EI",  # the acquisition function
            n_calls=iterations,  # the number of evaluations of f
            n_initial_points=20,  # the number of random initialization points
            random_state=42,  # the random seed
        )

        approx_x, approx_fn = expected_minimum(res)  # waht is this for?
        optimal_physics_param = np.array(approx_x).astype(np.float32)
        optimal_physics_param_list.append(optimal_physics_param)
    
    optimal_physics_param_list = np.stack(optimal_physics_param_list, axis=0)
    ppm_std = np.std(optimal_physics_param_list, axis=0)
    ppm_mean = np.mean(optimal_physics_param_list, axis=0)

    # rollout best ppm
    physics_param = ppm_mean.tolist()
    error = dynamics_error(physics_param, ppm_optimizer, state_init_list, state_real_list, actions)

    if return_res:
        return ppm_mean, error, init_error, res
    else:
        return ppm_mean, error, init_error


def optimize_cma(ppm_optimizer, actions, state_init_list, state_pred_list, state_real_list, 
                 num_optim_trials=1, iter_idx=0, iterations=50, return_res=False):

    pp_list_init = []
    physics_param_init = dict()
    for material_name, material_dim in ppm_optimizer.material_dims.items():
        pp_list_init.extend(ppm_optimizer.physics_param[material_name].tolist())
        physics_param_init[material_name] = ppm_optimizer.physics_param[material_name].unsqueeze(0)
    input_phys_dim = sum([material_dim for material_dim in ppm_optimizer.material_dims.values()])
    assert len(pp_list_init) == input_phys_dim

    init_error = dynamics_error(physics_param_init, ppm_optimizer, state_init_list, state_real_list, actions)    
    print(f"init error {init_error}")

    error_func = partial(
        dynamics_error,
        ppm_optimizer=ppm_optimizer, state_init_list=state_init_list, state_real_list=state_real_list, actions=actions,
    )

    std = 0.2
    print("init physics param", pp_list_init)
    print("std", std)

    optimal_physics_param_list = []
    for _ in range(num_optim_trials):
        es = cma.CMAEvolutionStrategy(pp_list_init, std, {'bounds': [-0.2, 1.2]})
        if iterations > 0:
            es.optimize(error_func, iterations=iterations)
        else:
            es.optimize(error_func)  # until stop
        res = es.result
        optimal_physics_param = np.array(res[0]).astype(np.float32)
        optimal_error = res[1]
        optimal_physics_param_list.append(optimal_physics_param)
    
    optimal_physics_param_list = np.stack(optimal_physics_param_list, axis=0)
    ppm_std = np.std(optimal_physics_param_list, axis=0)
    ppm_mean = np.mean(optimal_physics_param_list, axis=0)

    optimal_ppm = optimal_physics_param_list[0]
    print(f"optimal error {optimal_error}")

    # rollout best ppm
    physics_param = ppm_mean.tolist()
    error = dynamics_error(physics_param, ppm_optimizer, state_init_list, state_real_list, actions)
    print(f"optimal ppm rollout error {error}")

    if return_res:
        return optimal_ppm, optimal_error, init_error, es
    else:
        return optimal_ppm, optimal_error, init_error


def dynamics_error(physics_param, ppm_optimizer, state_init_list, state_real_list, actions):
    len_act = len(actions)
    physics_param = copy.deepcopy(physics_param)
    device = ppm_optimizer.device

    if isinstance(physics_param, list) or isinstance(physics_param, np.ndarray):
        assert len(list(ppm_optimizer.material_dims.keys())) == 1, 'only support single material now'
        material_name = list(ppm_optimizer.material_dims.keys())[0]
        physics_param = {material_name: torch.tensor(physics_param, dtype=torch.float32, device=device)}

    push_length = ppm_optimizer.task_config['push_length']
    max_nobj = ppm_optimizer.task_config['max_nobj']
    state_init_mask = np.zeros((len_act, max_nobj))
    state_final_mask = np.zeros((len_act, max_nobj))

    state_init_pad_list = []
    state_final_pad_list = []

    for i in range(len_act):
        state_init = state_init_list[i]
        state_final = state_real_list[i]

        state_init_mask[i, :state_init.shape[0]] = 1
        state_final_mask[i, :state_final.shape[0]] = 1

        state_init = np.pad(state_init, ((0, max_nobj - state_init.shape[0]), (0, 0)), mode='constant', constant_values=0)
        state_final = np.pad(state_final, ((0, max_nobj - state_final.shape[0]), (0, 0)), mode='constant', constant_values=0)
        
        state_init_pad_list.append(state_init)
        state_final_pad_list.append(state_final)
    
    state_init_pad = np.stack(state_init_pad_list, axis=0)
    state_final_pad = np.stack(state_final_pad_list, axis=0)

    state_init_all = torch.tensor(state_init_pad, dtype=torch.float32, device=device)
    state_init_mask = torch.tensor(state_init_mask, dtype=torch.bool, device=device)
    state_final_all = torch.tensor(state_final_pad, dtype=torch.float32, device=device)
    state_final_mask = torch.tensor(state_final_mask, dtype=torch.bool, device=device)

    actions = torch.from_numpy(np.stack(actions, axis=0)).to(device)

    out = dynamics_masked(state_init_all, state_init_mask, actions, ppm_optimizer.model, ppm_optimizer.device, ppm_optimizer, 
                          physics_param=physics_param)
    
    state_pred = out['state_seqs'].detach()  # .cpu().numpy()  # (bsz, max_nobj, 3)
    error = mean_chamfer(state_pred, state_final_all, state_init_mask, state_final_mask)

    mean_error = error.mean()
    return mean_error
