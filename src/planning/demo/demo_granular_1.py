import os
import argparse
import yaml
import torch
import numpy as np
import time
import glob

from planning.physics_param_optimizer import PhysicsParamOnlineOptimizer
from dynamics.gnn.model import DynamicsPredictor
from dynamics.utils import set_seed

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    args = arg_parser.parse_args()

    args.config = 'config/dynamics/granular.yaml'
    args.epoch = 100
    args.task_config = 'config/planning/granular.yaml'
    args.name = 'planning/dump/vis_demo/granular_1'

    with open(args.task_config, 'r') as f:
        task_config = yaml.load(f, Loader=yaml.CLoader)['task_config']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    epoch = args.epoch
    train_config = config['train_config']
    dataset_config = config['dataset_config']
    model_config = config['model_config']
    material_config = config['material_config']
    
    set_seed(train_config['random_seed'])

    checkpoint_dir = os.path.join(train_config['out_dir'], dataset_config['data_name'], 'checkpoints', 'model_{}.pth'.format(epoch))

    model = DynamicsPredictor(model_config, material_config, dataset_config, device)
    model.to(device)

    model.eval()
    model.load_state_dict(torch.load(checkpoint_dir, map_location='cpu'))

    save_dir = args.name
    material = task_config['material']
    ppm_optimizer = PhysicsParamOnlineOptimizer(task_config, model, material, device, save_dir)

    # optimize
    ppm_optimizer.optimize(9, iterations=50)
