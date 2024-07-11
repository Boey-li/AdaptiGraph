import numpy as np
import argparse
import yaml
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append('.')
from dynamics.gnn.model import DynamicsPredictor
from dynamics.dataset.dataset import DynDataset
from dynamics.utils import set_seed, dataloader_wrapper, grad_manager

def train(config):
    ## config
    dataset_config = config['dataset_config']
    train_config = config['train_config']
    model_config = config['model_config']
    material_config = config['material_config']
    
    data_name = dataset_config['data_name']
    out_dir = os.path.join(train_config['out_dir'], data_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)

    torch.autograd.set_detect_anomaly(True)
    set_seed(train_config['random_seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## load dataset
    n_future = dataset_config['n_future']
    phases = train_config['phases']
    
    datasets = {phase: DynDataset(
        dataset_config=dataset_config, 
        material_config=material_config,
        phase=phase
    ) for phase in phases}
    
    dataloaders = {phase: DataLoader(
        datasets[phase],
        batch_size=train_config['batch_size'],
        shuffle=(phase == 'train'),
        num_workers=train_config['num_workers']
    ) for phase in phases}
    
    dataloaders = {phase: dataloader_wrapper(dataloaders[phase], phase) for phase in phases}

    ## load model
    model = DynamicsPredictor(model_config, 
                              material_config, 
                              dataset_config,
                              device)
    model.to(device)

    mse_loss = torch.nn.MSELoss()
    loss_funcs = [(mse_loss, 1)]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    ## training
    loss_plot_list_train = []
    loss_plot_list_valid = [] 
    for epoch in range(train_config['n_epochs']):
        time1 = time.time()
        for phase in phases:
            with grad_manager(phase):
                if phase == 'train': 
                    model.train()
                else: 
                    model.eval()
                loss_sum_list = []
                n_iters = train_config['n_iters_per_epoch'][phase] \
                        if train_config['n_iters_per_epoch'][phase] != -1 else len(datasets[phase])
                for i in range(n_iters):
                    data = next(dataloaders[phase])
                    if phase == 'train':
                        optimizer.zero_grad()
                    data = {key: data[key].to(device) for key in data.keys()}
                    loss_sum = 0

                    future_state = data['state_future']  # (B, n_future, n_p, 3)
                    future_eef = data['eef_future']  # (B, n_future-1, n_p+n_s, 3)
                    future_action = data['action_future']  # (B, n_future-1, n_p+n_s, 3)

                    for fi in range(n_future):
                        gt_state = future_state[:, fi].clone()  # (B, n_p, 3)

                        pred_state, pred_motion = model(**data)

                        pred_state_p = pred_state[:, :gt_state.shape[1], :3].clone()
                        loss = [weight * func(pred_state_p, gt_state) for func, weight in loss_funcs]

                        loss_sum += sum(loss)

                        if fi < n_future - 1:
                            # build next graph
                            next_eef = future_eef[:, fi].clone()  # (B, n_p+n_s, 3)
                            next_action = future_action[:, fi].clone()  # (B, n_p+n_s, 3)
                            next_state = next_eef.unsqueeze(1)  # (B, 1, n_p+n_s, 3)
                            next_state[:, -1, :pred_state_p.shape[1]] = pred_state_p 
                            next_state = torch.cat([data['state'][:, 1:], next_state], dim=1)  # (B, n_his, n_p+n_s, 3)
                            data["state"] = next_state  # (B, n_his, N+M, state_dim)
                            data["action"] = next_action  # (B, N+M, state_dim) 

                    if phase == 'train':
                        loss_sum.backward()
                        optimizer.step()
                        if i % train_config['log_interval'] == 0:
                            print(f'Epoch {epoch}, iter {i}, loss {loss_sum.item()}')
                            loss_sum_list.append(loss_sum.item())
                    if phase == 'valid':
                        loss_sum_list.append(loss_sum.item())

                if phase == 'valid':
                    print(f'\nEpoch {epoch}, valid loss {np.mean(loss_sum_list)}')

                if phase == 'train':
                    loss_plot_list_train.append(np.mean(loss_sum_list))
                if phase == 'valid':
                    loss_plot_list_valid.append(np.mean(loss_sum_list))
        
        if ((epoch + 1) < 100 and (epoch + 1) % 10 == 0) or (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, 'checkpoints', f'model_{(epoch + 1)}.pth'))
        torch.save(model.state_dict(), os.path.join(out_dir, 'checkpoints', f'latest.pth'))
        torch.save(optimizer.state_dict(), os.path.join(out_dir, 'checkpoints', f'latest_optim.pth'))

        # plot figures
        plt.figure(figsize=(20, 5))
        plt.plot(loss_plot_list_train, label='train')
        plt.plot(loss_plot_list_valid, label='valid')
        # cut off figure
        ax = plt.gca()
        y_min = min(min(loss_plot_list_train), min(loss_plot_list_valid))
        y_min = min(loss_plot_list_valid)
        y_max = min(3 * y_min, max(max(loss_plot_list_train), max(loss_plot_list_valid)))
        ax.set_ylim([0, y_max])
        # save
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'loss.png'), dpi=300)
        plt.close()

        time2 = time.time()
        print(f'Epoch {epoch} time: {time2 - time1}\n')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, default='config/dynamics/rope.yaml')
    args = arg_parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    train(config)
