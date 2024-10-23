import os
import numpy as np
import time
import multiprocessing as mp
import argparse
import pickle

import sys
sys.path.append(".")

from sim.sim_env.flex_env import FlexEnv
from sim.data_gen.data import store_data
from sim.utils import load_yaml

# data generation
def gen_data(info):
    start_time = time.time()
    
    idx_episode = info["epi"]
    save_data = info["save_data"]
    
    if save_data:
        # create folder
        obj_dir = os.path.join(data_dir, obj+"_set_action_trial2")
        epi_dir = os.path.join(obj_dir, f'{idx_episode:06}')
        os.makedirs(epi_dir, exist_ok=True)

    # set env 
    env = FlexEnv(config)
    np.random.seed(idx_episode)
    print('episode start:', idx_episode)
    
    # reset env
    # data: [imgs_list, particle_pos_list, eef_states_list]
    data = env.reset(save_data)
    
    # get physics params
    physics_params = env.get_property_params()
    print(f'Episode {idx_episode} physics params: {physics_params}')
    
    # create actions
    actions = np.zeros((n_timestep, action_dim))
    
    # save initial data
    if save_data:
        # save data [info, action, positions, eef_states, observations]
        filename = os.path.join(epi_dir, f'new_sim{0:02}.h5')
        store_data(filename, data, actions[0])
        # save physics params
        with open(os.path.join(epi_dir, 'new_sim_property_params.pkl'), 'wb') as f:
            pickle.dump(physics_params, f)
        # save camera params
        if idx_episode == base_0:
            cam_dir = os.path.join(obj_dir, 'cameras')
            os.makedirs(cam_dir, exist_ok=True)
            # cam_intrinsic_params: (num_cameras, 4)
            # cam_extrinsic_matrix: (num_cameras, 4, 4)
            cam_intrinsic_params, cam_extrinsic_matrix = env.cam_intrinsic_params, env.cam_extrinsic_matrix
            np.save(os.path.join(cam_dir, 'new_sim_intrinsic.npy'), cam_intrinsic_params)
            np.save(os.path.join(cam_dir, 'new_sim_extrinsic.npy'), cam_extrinsic_matrix)
        
    # n_pushes
    color_threshold = dataset_config['color_threshold']
    img = env.render()
    last_img = img.copy()
    stuck = False
    for idx_timestep in range(n_timestep):
        color_diff = 0
        data = [], [], [] # reinitialize data for each timestep
        for k in range(10):
            u = None
            
            if obj in ['cloth']:
                if idx_timestep == 0:
                    u, boundary_points, boundary = env.sample_action(init=True)
                else:
                    u, boundary_points, boundary = env.sample_action(boundary_points=boundary_points, boundary=boundary)
            else:
                #u = env.sample_action() # [x_start, z_start, x_end, z_end]
                # hard set the start and end to be a specific action
                print("data gen, hard set action")
                if idx_timestep == 0:
                    u = np.array([0.1, 0.0, 0.3, 0.0])
                elif idx_timestep == 1:
                    u = np.array([0.3, 0.0, -0.2, 0.0])
                elif idx_timestep == 2:
                    u = np.array([-0.2, 0.0, -0.2, 0.1])
                elif idx_timestep == 3:
                    u = np.array([-0.2, 0.1, 0.0, 0.05])
                elif idx_timestep == 4:
                    u = np.array([0.0, 0.05, 0.2, 0.3])
                else:
                    u = np.array([0.2, 0.3, 0.0, 0.0])
            # write out to file the action
            np.save(os.path.join(epi_dir, f'action_{idx_timestep:02}.npy'), u)
            
            if u is None:
                stuck = True
                print(f"Episode {idx_episode} timestep {idx_timestep}: No valid action found!")
                break
    
            # step
            img, data = env.step(u, save_data, data)
            
            # check valid/invalid action to make difference large enough
            color_diff = np.mean(np.abs(img[:, :, :3] - last_img[:, :, :3]))
            if color_diff < color_threshold:
                data = [], [], []
                if k == 9:
                    stuck = True
                    print('The process is stucked on episode %d timestep %d!!!!' % (idx_episode, idx_timestep))
            else:
                break
        
        # save action
        if not stuck:
            actions[idx_timestep] = u
            last_img = img.copy()
            if save_data:
                filename = os.path.join(epi_dir, f'{idx_timestep+1:02}.h5')
                store_data(filename, data, actions[idx_timestep])
                print('episode %d timestep %d done!!! step: %d' % (idx_episode, idx_timestep, env.count))
        else:
            break       
        
    end_time = time.time()
    print('Episode %d time: ' % idx_episode, end_time - start_time, '\n')
            
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/data_gen/rope.yaml')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    
    # load config
    config = load_yaml(args.config)
    dataset_config = config['dataset']
    data_dir = dataset_config['folder']
    os.system("mkdir -p %s" % data_dir)
    
    obj = dataset_config['obj']
    
    base_0 = dataset_config['base']
    n_worker = dataset_config['n_worker']
    n_episode = dataset_config['n_episode']
    n_timestep = dataset_config['n_timestep']

    action_dim = dataset_config['action_dim']
    cam_view = dataset_config['camera_view']

    if args.debug:
        info = {
            "epi": base_0,
            "save_data": args.save,
        }
        gen_data(info)
    else:
        ### multiprocessing
        num_bases = n_episode // n_worker
        bases = [base_0 + n_worker*n for n in range(num_bases)]
        print(f"num_bases: {len(bases)}")
        print(bases)

        for base in bases:
            print("base:", base)
            infos=[]
            for i in range(n_worker):
                info = {
                    "epi": base+i,
                    "save_data": args.save,
                }
                infos.append(info)
            pool = mp.Pool(processes=n_worker)
            pool.map(gen_data, infos)
