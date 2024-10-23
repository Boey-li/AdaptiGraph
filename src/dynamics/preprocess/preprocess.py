import os
import glob
import numpy as np
import argparse
import pickle
import time

import sys
sys.path.append('.')
from sim.utils import load_yaml
from sim.data_gen.data import load_data
from dynamics.utils import quaternion_to_rotation_matrix

"""
Preprocess:
    - frame pairs
    - eef and object positions
    - physics params
    - metadata
"""
    
def process_eef(eef_states, eef_dataset):
    """
    eef_states: (T, N_eef, 14)
    """
    T = eef_states.shape[0]
    if len(eef_states.shape) == 2:
        eef_states = eef_states.reshape(T, 1, 14)
    eef_pos = eef_dataset['pos']
    N_eef = len(eef_pos)

    out_eefs = np.zeros((T, eef_dataset['max_neef'], 3))   
    assert N_eef == eef_dataset['max_neef'], 'Number of eef not match.' 
    
    # process eef
    for i in range(T):
        for j in range(N_eef):
            if j >= eef_states.shape[1]:
                # granular case
                eef_idx = eef_states.shape[1] - 1
            else:
                eef_idx = j
            eef_state = eef_states[i][eef_idx]
            eef_pos_0 = eef_state[0:3]
            eef_quat = eef_state[6:10]
            eef_rot = quaternion_to_rotation_matrix(eef_quat)
            eef_final_pos = eef_pos_0 + np.dot(eef_rot, eef_pos[j])
            out_eefs[i, j] = eef_final_pos
    return out_eefs

def extract_physics(physics_path, obj):
    with open(physics_path, 'rb') as f:
        properties = pickle.load(f)
    # extract physics params
    if obj == 'rope':
        phys_param = np.array([
            properties['stiffness']
        ]).astype(np.float32)
    elif obj == 'granular':
        phys_param = np.array([
            properties['granular_scale']
        ])
    elif obj == 'cloth':
        phys_param = np.array([
            properties['sf']
        ])
    else:
        raise ValueError('Invalid object type.')
    return phys_param

def extract_push(eef, dist_thresh, n_his, n_future, n_frames):
    """
    eef: (T, N_eef, 3)
    """
    T = eef.shape[0]
    eef = eef[:, 0] # (T, 3)
    
    # generate start-end pair
    frame_idxs = []
    cnt = 0
    start_frame = 0
    end_frame = T
    for fj in range(T):
        curr_frame = fj
        
        # search backward (n_his)
        eef_curr = eef[curr_frame]
        frame_traj = [curr_frame]
        fi = fj
        while fi >= start_frame:
            eef_fi = eef[fi]
            x_curr, z_curr = eef_curr[0], eef_curr[2]
            x_fi, z_fi = eef_fi[0], eef_fi[2]
            dist_curr = np.sqrt((x_curr - x_fi) ** 2 + (z_curr - z_fi) ** 2)
            if dist_curr >= dist_thresh:
                frame_traj.append(fi)
                eef_curr = eef_fi
            fi -= 1
            if len(frame_traj) == n_his:
                break
        else:
            # pad to n_his
            frame_traj = frame_traj + [frame_traj[-1]] * (n_his - len(frame_traj))
        frame_traj = frame_traj[::-1]
        
        # search forward (n_future)
        eef_curr = eef[curr_frame]
        fi = fj
        while fi < end_frame:
            eef_fi = eef[fi]
            x_curr, z_curr = eef_curr[0], eef_curr[2]
            x_fi, z_fi = eef_fi[0], eef_fi[2]
            dist_curr = np.sqrt((x_curr - x_fi) ** 2 + (z_curr - z_fi) ** 2)
            if dist_curr >= dist_thresh:
                frame_traj.append(fi)
                eef_curr = eef_fi
            fi += 1
            if len(frame_traj) == n_his + n_future:
                cnt += 1
                break
        else:
            # pad to n_future
            frame_traj = frame_traj + [frame_traj[-1]] * (n_his + n_future - len(frame_traj))
            cnt += 1
        
        frame_idxs.append(frame_traj)
        
        # push centered
        if fj == end_frame - 1:
            frame_idxs = np.array(frame_idxs)
            frame_idxs = frame_idxs + n_frames # add previous steps
    
    return frame_idxs, cnt

def preprocess(config):
    time_start = time.time()
    
    # config
    dataset_config = config['dataset_config']
    data_name = dataset_config['data_name']
    eef_dataset = dataset_config['eef']
    
    data_dir = os.path.join(dataset_config['data_dir'], data_name+"_set_action_first_try")
    save_dir = os.path.join(dataset_config['prep_data_dir'], data_name+"_set_action_first_try")
    push_save_dir = os.path.join(save_dir, 'frame_pairs')
    os.makedirs(push_save_dir, exist_ok=True)
    
    n_his = dataset_config['n_his']
    n_future = dataset_config['n_future']
    dist_thresh = dataset_config['dist_thresh']    
    
    # episodes
    epi_list = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f.isdigit()])
    num_epis = len(epi_list)
    print(f"Preprocessing starts. Number of episodes: {num_epis}.")
    
    # preprocessing
    all_eef_pos = [] # (n_epis, N_eef, 3) 
    all_obj_pos = [] # (n_epis, N_obj, 3)
    phys_params = [] # (n_epis, N_phy, 1)
    for epi_idx, epi in enumerate(epi_list):
        epi_time_start = time.time()
        
        epi_dir = os.path.join(data_dir, epi)
        
        # preprocess property params
        physics_path = os.path.join(epi_dir, 'property_params.pkl')
        phys_param = extract_physics(physics_path, data_name)
        phys_params.append(phys_param)
        
        # preprocess step info
        num_steps = len(list(glob.glob(os.path.join(epi_dir, '*.h5')))) - 1
        
        eef_steps, obj_steps = [], []
        n_frames = 0
        for step_idx in range(1, num_steps+1):
            # extract data
            data_path = os.path.join(epi_dir, f'{step_idx:02}.h5')
            data = load_data(data_path) # ['action', 'eef_states', 'info', 'observations', 'positions']
            
            eef_states = data['eef_states'] # (T, N_eef, 14)
            positions = data['positions'] # (T, N_obj, 3)
            
            # preprocess eef and push
            out_eef = process_eef(eef_states, eef_dataset) # (T, N_eef, 3)
            frame_idxs, cnt = extract_push(out_eef, dist_thresh, n_his, n_future, n_frames)
            assert len(frame_idxs) == cnt, 'Number of pushes not match.'
            n_frames += cnt
            
            # eef and object positions
            eef_steps.append(out_eef)
            obj_steps.append(positions)
            
            # save frame idxs
            np.savetxt(os.path.join(push_save_dir, f'{epi}_{(step_idx):02}.txt'), frame_idxs, fmt='%d')
            print(f"Preprocessed episode {epi_idx+1}/{num_epis}, step {step_idx}/{num_steps}: Number of pushes {cnt}.")
        
        eef_steps = np.concatenate(eef_steps, axis=0)
        obj_steps = np.concatenate(obj_steps, axis=0)
        all_eef_pos.append(eef_steps)
        all_obj_pos.append(obj_steps)
        assert eef_steps.shape[0] == obj_steps.shape[0] == n_frames
        
        epi_time_end = time.time()
        print(f'Episode {epi_idx+1}/{num_epis} has frames {obj_steps.shape[0]} took {epi_time_end - epi_time_start:.2f}s.')
    
    # save physics params
    phys_params = np.stack(phys_params, axis=0)
    phys_params_max = np.max(phys_params, axis=0)
    phys_params_min = np.min(phys_params, axis=0)
    phys_params_range = np.stack([phys_params_min, phys_params_max], axis=0)
    print(f"Physics params range: {phys_params_range}")
    np.savetxt(os.path.join(save_dir, 'phys_range.txt'), phys_params_range)
    
    # save eef and object positions
    pos_path = os.path.join(save_dir, 'positions.pkl')
    pos_info = {
        'eef_pos': all_eef_pos, 
        'obj_pos': all_obj_pos,
    }
    with open(pos_path, 'wb') as f:
        pickle.dump(pos_info, f)
    assert len(all_eef_pos) == len(all_obj_pos) == num_epis
    
    # save metadata
    with open(os.path.join(save_dir, 'metadata.txt'), 'w') as f:
        f.write(f'{dist_thresh},{n_future},{n_his}')
    
    time_end = time.time()
    print(f"Preprocessing finished for Episodes {num_epis}. Time taken: {time_end - time_start:.2f}s.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/dynamics/rope.yaml')
    args = parser.parse_args()
    
    config = load_yaml(args.config)
    
    preprocess(config)