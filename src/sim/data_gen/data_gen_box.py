import os
import cv2
import time
import numpy as np
import argparse

import pygame
from sim.sim_env.pymunk_env import BoxSim
from sim.utils import rand_float, load_yaml

def convert_coordinates(point, screen_height):
        return np.array([point[0], screen_height - point[1]])

def gen_box_data(config, save_dir, epi_idx):
    start_time = time.time()
    np.random.seed(epi_idx)
    
    # create folder
    out_dir = os.path.join(save_dir, f"{epi_idx:06}")
    os.makedirs(out_dir, exist_ok=True)
    
    # set env
    screen_width = config['screenWidth']
    screen_height = config['screenHeight']
    
    box_width = rand_float(*config['box_width'])
    box_height = rand_float(*config['box_height'])
    sim = BoxSim(screen_width, screen_height, box_width, box_height)

    # center of mass and friction
    box_size = sim.get_obj_size()
    
    center_of_mass = (rand_float(-box_size[0]/2, box_size[0]/2), rand_float(-box_size[1]/2, box_size[1]/2))
    friction = 0.5
    sim.add_box(center_of_mass, friction)
    print(f"Episode {epi_idx}, center of mass: {center_of_mass}, friction: {friction}")

    # init pos for pusher
    box_pos = sim.get_obj_state()[:2]
    box_center = np.array([box_pos[0] - center_of_mass[0], box_pos[1] - center_of_mass[1]])
    print("box init pos: ", box_pos)
    pusher_choice = np.random.choice([0, 1, 2, 3])
    if pusher_choice == 0: # top to bottom
        pusher_x = rand_float(box_center[0] - box_size[0] / 2, box_center[0] + box_size[0] / 2) 
        pusher_y = box_center[1] + box_size[1] / 2 + rand_float(100, 200)
    elif pusher_choice == 1: # bottom to top
        pusher_x = rand_float(box_center[0] - box_size[0] / 2, box_center[0] + box_size[0] / 2)
        pusher_y = box_center[1] - box_size[1] / 2 - rand_float(100, 200)
    elif pusher_choice == 2: # left to right
        pusher_x = box_center[0] - box_size[0] / 2 - rand_float(100, 200)
        pusher_y = rand_float(box_center[1] - box_size[1] / 2, box_center[1] + box_size[1] / 2)
    elif pusher_choice == 3: # right to left
        pusher_x = box_center[0] + box_size[0] / 2 + rand_float(100, 200)
        pusher_y = rand_float(box_center[1] - box_size[1] / 2, box_center[1] + box_size[1] / 2)

    pusher_pos = (pusher_x, pusher_y)
    n_iter_rest = 100
    for i in range(n_iter_rest):
        sim.update(pusher_pos)

    n_sim_step = config['n_timestep']
    box_states = []
    eef_states = []
    for i in range(n_sim_step):
        pusher_x, pusher_y = pusher_pos
        
        if pusher_choice == 0: # top to bottom
            pusher_y -= 10
        elif pusher_choice == 1: # bottom to top
            pusher_y += 10
        elif pusher_choice == 2: # left to right
            pusher_x += 10
        elif pusher_choice == 3: # right to left
            pusher_x -= 10
            
        pusher_pos = (pusher_x, pusher_y)
        sim.update(pusher_pos)
        
        # save image
        img_out_dir = os.path.join(out_dir, "images")
        os.makedirs(img_out_dir, exist_ok=True)
        out_path = os.path.join(img_out_dir, f"{i:03d}.png")
        sim.save_image(out_path)
        
        # save info
        box_init_state = sim.get_obj_state()[:3] # (x, y, theta)
        box_state = np.array([box_init_state[0], box_init_state[1], box_init_state[2]])
        # box_state[:2] = convert_coordinates(box_state[:2], screen_height)
        # eef_state = convert_coordinates(np.array(pusher_pos), screen_height)
        box_state[:2] = box_state[:2]
        eef_state = np.array(pusher_pos)
        box_states.append(box_state)
        eef_states.append(eef_state)
        
        time.sleep(0.1)
    
    
    np.save(os.path.join(out_dir, "box_states.npy"), np.array(box_states))
    np.save(os.path.join(out_dir, "eef_states.npy"), np.array(eef_states))
    # Save center of mass and friction and box size
    box_com = np.array([
        [box_size[0], box_size[1]],
        [center_of_mass[0], center_of_mass[1]],
    ])
    np.save(os.path.join(out_dir, "box_com.npy"), box_com)
    
    end_time = time.time()
    print(f"Episode {epi_idx} finshed!!! Time: {end_time - start_time}")

    sim.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/data_gen/box.yaml')
    args = parser.parse_args()
    
    # load config
    config = load_yaml(args.config)
    dataset_config = config['dataset']
    obj = dataset_config['obj']
    save_dir = os.path.join(dataset_config['folder'], obj)
    os.makedirs(save_dir, exist_ok=True)
    
    base = dataset_config['base']
    n_episode = dataset_config['n_episode']
    
    for epi_idx in range(base, base + n_episode):
        gen_box_data(dataset_config, save_dir, epi_idx)
        
