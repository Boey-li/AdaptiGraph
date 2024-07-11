import os
import glob
import numpy as np
import pickle

def load_pairs(pairs_path, episode_range):
    pair_lists = []
    for episode_idx in episode_range:
        n_pushes = len(list(glob.glob(os.path.join(pairs_path, f'{episode_idx:06}_*.txt'))))
        for push_idx in range(1, n_pushes+1):
            frame_pairs = np.loadtxt(os.path.join(pairs_path, f'{episode_idx:06}_{push_idx:02}.txt'))
            if len(frame_pairs.shape) == 1: continue
            episodes = np.ones((frame_pairs.shape[0], 1)) * episode_idx
            pairs = np.concatenate([episodes, frame_pairs], axis=1) # (T, 8)
            pair_lists.extend(pairs)
    pair_lists = np.array(pair_lists).astype(int)
    return pair_lists

def load_dataset(dataset_config, material_config, phase='train'):
    # config
    data_name = dataset_config['data_name']
    data_dir = os.path.join(dataset_config['data_dir'], data_name)
    prep_dir = os.path.join(dataset_config['prep_data_dir'], data_name)
    ratio = dataset_config['ratio']
    
    # episodes
    num_epis = len(sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f.isdigit()]))
    print(f"Found number of episodes: {num_epis}.")
    
    # load data pairs
    episode_range_phase = range(
        int(num_epis * ratio[phase][0]),
        int(num_epis * ratio[phase][1])
    )
    pairs_path = os.path.join(prep_dir, 'frame_pairs')
    pair_lists = load_pairs(pairs_path, episode_range_phase)
    print(f'{phase} dataset has {len(list(episode_range_phase))} episodes, {len(pair_lists)} frame pairs')
    
    # load physics params
    physics_params = []
    for episode_idx in range(num_epis):
        physics_path = os.path.join(data_dir, f"{episode_idx:06}/property_params.pkl")
        with open(physics_path, 'rb') as f:
            properties = pickle.load(f)
        
        physics_params_episode = {}
        for material_name in dataset_config["materials"]:
            material_params = material_config[material_name]['physics_params']

            phys_norm_max = 1.0
            phys_norm_min = 0.0
            
            used_params = []
            for item in material_params:
                if item['name'] in properties.keys() and item['use']:
                    range_min = item['min']
                    range_max = item['max']
                    used_params.append((properties[item['name']] - range_min) / (range_max - range_min + 1e-6))
            
            used_params = np.array(used_params).astype(np.float32)
            used_params = used_params * (phys_norm_max - phys_norm_min) + phys_norm_min
            physics_params_episode[material_name] = used_params
        
        physics_params.append(physics_params_episode)
    
    return pair_lists, physics_params

def load_positions(dataset_config):
    ## config
    data_name = dataset_config['data_name']
    prep_dir = os.path.join(dataset_config['prep_data_dir'], data_name)
    
    ## load positions
    # ['eef_pos', 'obj_pos', 'phys_params']
    # eef_pos: (n_epis, T, N_eef, 3)
    # obj_pos: (n_epis, T, N_obj, 3)
    # phys_params: (n_epis, 1)
    positions_path = os.path.join(prep_dir, 'positions.pkl')
    with open(positions_path, 'rb') as f:
        positions = pickle.load(f) 
    eef_pos = positions['eef_pos'] 
    obj_pos = positions['obj_pos']
    return eef_pos, obj_pos
