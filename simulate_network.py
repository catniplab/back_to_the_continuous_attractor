# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 08:36:51 2025

@author: abel_
"""
import os
import torch
import glob
import numpy as np
import pandas as pd

from load_network import get_tr_par
from tasks import angularintegration_task, angularintegration_task_constant

def simulate_rnn(net, task, T, batch_size):
    input, target, mask = task(batch_size); input = torch.from_numpy(input).float();
    output, trajectories = net(input, return_dynamics=True); 
    output = output.detach().numpy();
    trajectories = trajectories.detach().numpy()
    return input, target, mask, output, trajectories

def simulate_rnn_with_input(net, input, h_init):
    input = torch.from_numpy(input).float();
    output, trajectories = net(input, return_dynamics=True, h_init=h_init); 
    output = output.detach().numpy();
    trajectories = trajectories.detach().numpy()
    return output, trajectories

def simulate_rnn_with_task(net, task, T, h_init, batch_size=256):
    input, target, mask = task(batch_size);
    input_ = torch.from_numpy(input).float();
    target_ = torch.from_numpy(target).float();
    output, trajectories = net(input_, return_dynamics=True, h_init=h_init, target=target_); 
    
    output = output.detach().numpy();
    trajectories = trajectories.detach().numpy()
    return input, target, mask, output, trajectories
  
def test_network(net,T=25.6, dt=.1, batch_size=4, from_t=0, to_t=None, random_seed=100):
    #test network on angular integration task
    np.random.seed(random_seed)
    task = angularintegration_task(T=T, dt=dt, sparsity=1, random_angle_init='equally_spaced');
    #task = angularintegration_task_constant(T=T, dt=dt, speed_range=[0.1,0.1], sparsity=1, random_angle_init='equally_spaced');
    input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, T, 'random', batch_size=batch_size)
    target_power = np.mean(target[:,from_t:to_t,:]**2)
    mse = np.mean((target[:,from_t:to_t,:] - output[:,from_t:to_t,:])**2)
    mse_normalized = mse/target_power
    return mse, mse_normalized

def get_autonomous_dynamics(net, T=128, dt=.1, batch_size=32):
    task = angularintegration_task_constant(T=T, dt=dt, speed_range=[0.,0.], sparsity=1, random_angle_init='equally_spaced');
    input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, T, 'random', batch_size=batch_size)
    return output, trajectories

def get_autonomous_dynamics_from_hinit(net, h_init, T=128):
    input = np.zeros((h_init.shape[0], T, net.dims[0]))
    output, trajectories = simulate_rnn_with_input(net, input, h_init=h_init)
    return output, trajectories
    

def test_networks_in_folder(folder, df=None):
    exp_list = glob.glob(folder + "/res*")
    nexps = len(exp_list)

    for exp_i in range(nexps):
        path = exp_list[exp_i]
        print(path)        
        net, result = load_net_path(path)
        mse = test_network(net)
        training_kwargs = result['training_kwargs']
        T, N, I, S, R, M, clip_gradient = get_tr_par(training_kwargs)
        df = df.append({'T': T, 'N': N, 'I': I, 'S': S, 'R': R, 'M': M, 'clip_gradient':clip_gradient,
                        'trial': exp_i,
                        'mse': mse}, ignore_index=True)
    return df

def test_all_networks(folder):
    df = pd.DataFrame(columns=['T', 'N', 'I', 'S', 'R', 'M', 'trial', 'mse'])

    for dirName, subdirList, fileList in os.walk(folder):
        print(dirName)
        df = test_networks_in_folder(dirName, df=df)

    return df

