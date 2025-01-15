# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:53:54 2024

@author: 
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from sklearn.cluster import DBSCAN

from tasks import angularintegration_task, angularintegration_task_constant, double_angularintegration_task

from lstm import vf_norm_from_outtraj, mean_fp_distance, boa, find_fixed_point, nmse, angluar_error
from double_angular_analysis import double_mean_fp_distance, double_boa, angular_double_error
from analysis_functions import calculate_lyapunov_spectrum, participation_ratio, identify_limit_cycle, find_periodic_orbits, find_analytic_fixed_points, powerset, decibel
from utils import makedirs


epsilon=.1
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

tanh = nn.Tanh()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.5, init_weight_radius_scaling=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_to_hidden = nn.Parameter(torch.Tensor(output_size, hidden_size))
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weight_radius_scaling = init_weight_radius_scaling
        self._initialize_weights()
    
    def _initialize_weights(self):
        stdv = self.init_weight_radius_scaling / np.sqrt(self.hidden_size)
        for name, param in self.gru.named_parameters():
            if 'weight_hh' in name:  # Only initialize recurrent weights
                nn.init.uniform_(param, -stdv, stdv)

    def forward(self, x, target):
        batch_size = x.shape[0]
        h_init_torch = nn.Parameter(torch.Tensor(self.num_layers, batch_size, self.hidden_size)).to(x.device)
        h_init_torch.requires_grad = False
        h_init_torch = target[:, 0, :].matmul(self.output_to_hidden)
        h_init_torch = tanh(h_init_torch)
        h0 = h_init_torch.unsqueeze(0).expand(self.num_layers, -1, -1)  # (num_layers, batch_size, hidden_size)
        out, hn = self.gru(x, h0)
        out = self.dropout(out)
        out = self.fc(out)
        return out, hn
    
    def sequence(self, x, target):
        batch_size = x.shape[0]
        h_init_torch = target[:, 0, :].matmul(self.output_to_hidden)
        h_init_torch = tanh(h_init_torch)
        h0 = h_init_torch.unsqueeze(0).expand(self.num_layers, batch_size, self.hidden_size).contiguous()
        all_out = []
        all_hidden_states = []
        seq_len = x.size(1)  # Assuming x is of shape (batch_size, seq_len, input_size)
        for t in range(seq_len):
            xt = x[:, t, :].unsqueeze(1)
            out, h0 = self.gru(xt, h0)
            all_out.append(out)
            all_hidden_states.append(h0[-1].unsqueeze(1))
        all_out = torch.cat(all_out, dim=1)
        all_hidden_states = torch.cat(all_hidden_states, dim=1)
        return all_out, all_hidden_states

def extract_gru_parameters(model):
    hidden_size = model.hidden_size
    params = {}
    params['W_r'] = model.gru.weight_ih_l0[:hidden_size, :].detach().numpy()
    params['W_z'] = model.gru.weight_ih_l0[hidden_size:2*hidden_size, :].detach().numpy()
    params['W_h'] = model.gru.weight_ih_l0[2*hidden_size:, :].detach().numpy()
    params['U_r'] = model.gru.weight_hh_l0[:hidden_size, :].detach().numpy()
    params['U_z'] = model.gru.weight_hh_l0[hidden_size:2*hidden_size, :].detach().numpy()
    params['U_h'] = model.gru.weight_hh_l0[2*hidden_size:, :].detach().numpy()
    params['b_r'] = model.gru.bias_ih_l0[:hidden_size].detach().numpy() + model.gru.bias_hh_l0[:hidden_size].detach().numpy()
    params['b_z'] = model.gru.bias_ih_l0[hidden_size:2*hidden_size].detach().numpy() + model.gru.bias_hh_l0[hidden_size:2*hidden_size].detach().numpy()
    params['b_h'] = model.gru.bias_ih_l0[2*hidden_size:].detach().numpy() + model.gru.bias_hh_l0[2*hidden_size:].detach().numpy()
    return params

def test_model(model, task, batch_size):
    with torch.no_grad():
        inputs, targets, mask = task(batch_size)
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        targets = torch.tensor(targets, dtype=torch.float32).to(device)
        outputs, _ = model(inputs, targets)
        _, hs = model.sequence(inputs, targets)
    targets = targets.detach().numpy()
    outputs = outputs.detach().numpy()
    hs = hs.detach().numpy()
    hs = hs.squeeze()
    trajectories = hs
    from_t = 0
    to_t = None
    target_power = np.mean(targets[:, from_t:to_t, :] ** 2)
    mse = np.mean((targets[:, from_t:to_t, :] - outputs[:, from_t:to_t, :]) ** 2)
    mse_normalized = mse / target_power
    return decibel(mse_normalized), outputs, trajectories



    return inputs, targets, outputs, trajectories

def train_model(model, task, num_epochs=100, batch_size=32, learning_rate=0.001, 
                clip_norm=0., output_noise_level=0.01, weight_decay=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
                            {'params': model.gru.parameters(), 'weight_decay': weight_decay},
                            {'params': model.fc.parameters(), 'weight_decay': 0.0},
                            {'params': model.output_to_hidden, 'weight_decay': 0.0}
                        ], lr=learning_rate)
    
    model.to(device)
    criterion.to(device)

    # for epoch in range(num_epochs):
    epoch = 0
    losses = [] 
    while epoch < num_epochs:

        inputs, targets, mask = task(batch_size)
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        targets = torch.tensor(targets, dtype=torch.float32).to(device)
        output_noise = torch.randn(batch_size, inputs.shape[1], targets.shape[-1]).to(device) * output_noise_level
        targets += output_noise
        mask = torch.tensor(mask, dtype=torch.float32).to(device)
        
        # Save model and optimizer state
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()

        outputs, _ = model(inputs, targets)
        
        # Check for inf values in outputs
        if torch.isinf(outputs).any():
            print(f'Inf detected in outputs at epoch {epoch}. Scaling down weights.')
            with torch.no_grad():
                for param in model.parameters():
                    param *= model.init_weight_radius_scaling
        
        loss = criterion(outputs * mask, targets * mask)
        
        # Check for NaNs in loss
        if torch.isnan(loss):
            return [False]
            # if epoch == 0:
            #     print(f'NaN detected in loss at epoch {epoch}. Reinitializing model parameters.')
            #     model._initialize_weights()
            #     optimizer = optim.Adam([
            #                 {'params': model.gru.parameters(), 'weight_decay': weight_decay},
            #                 {'params': model.fc.parameters(), 'weight_decay': 0.0},
            #                 {'params': model.output_to_hidden, 'weight_decay': 0.0}
            #             ], lr=learning_rate)
            #     optimizer.zero_grad()

                
            # else:
                # print(f'NaN detected in loss at epoch {epoch}. Rolling back to previous state.')
                # model.load_state_dict(model_state_dict)
                # optimizer.load_state_dict(optimizer_state_dict)
                # with torch.no_grad():
                #     for param in model.parameters():
                #         param *= model.init_weight_radius_scaling
                # optimizer.zero_grad()
                # epoch+=1
                # continue
        
        optimizer.zero_grad()
        loss.backward()
        if clip_norm > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        
        losses.append(loss.item())  # Save the loss value

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        epoch+=1
        
    losses_array = np.array(losses)
    return losses_array

def train_n(n, task, input_size=1, hidden_size=64, output_size=2,
            num_epochs=5000, batch_size=64, learning_rate=0.001, clip_norm=1, dropout=0.,
            exp_path='C:/Users/abel_/Documents/Lab/projects/topological_diversity/experiments/angular_integration_old/N128_T128_noisy/gru/'):
    makedirs(exp_path)
    i=0
    while i<n:
        model = GRUModel(input_size, hidden_size, output_size, dropout=0.5, init_weight_radius_scaling=0.25)
        losses = train_model(model, task, num_epochs=5000, batch_size=batch_size, output_noise_level=0.01,
                             weight_decay=0.0001, learning_rate=0.01, clip_norm=100);
        if not losses.any():
            continue
        torch.save(model.state_dict(), exp_path+f'/model_{i}.pth')
        np.save(exp_path+f'/losses_{i}.npy', losses)
        i+=1
        
        
        
        
# fig, ax = plt.subplots(1, 1, figsize=(5, 3));
# N_color_dict = {64:'b', 128:'g', 256:'orange'}
# for N in [64,128,256]:
#     df_N = df[df['N'] == N]
#     error_T1 = np.array(df_N['mean_error_0'].tolist())[:,127]
#     ax.plot(df_N['vf_infty'], error_T1, 'X', color=N_color_dict[N])
#     error_10T1 = np.array(df_N['mean_error_0'].tolist())[:,-1]
#     ax.plot(df_N['vf_infty'], error_10T1, 'X', color=N_color_dict[N], fillstyle='none')

#     #ax.plot(df_N['nfps'], df_N['mean_fp_dist'], '.', color='magenta')
# varphis = np.linspace(0,0.04)
# ax.plot(varphis, 12.8*varphis, 'r--')
# ax.set_xlim([0,.04])
# ax.set_ylim([0,1.25])

# lines = [Line2D([0], [0], color=c, linewidth=2, linestyle='-') for c in ['b', 'g', 'orange']]
# first_legend = plt.legend(lines, N_color_dict.keys(), title='Network Size', loc='upper right')
# ax.add_artist(first_legend)

# plt.xlabel(r"$\|\varphi\|_\infty$"); plt.ylabel("angular error (rad)");
#plt.savefig('C:\\Users\\abel_\\Documents\\Lab\\Projects\\topological_diversity/experiments/angular_integration_old/gru_vfnorm_vs_mae.pdf');

        
        
def grid_search(task, input_size=1, hidden_size=64, output_size=2,
                num_epochs=5000, batch_size=64, learning_rate=0.001, dropout=0.,
                exp_path=''):
    
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    scale_factors = [0.8, .9, 0.99, 1.]
    weight_decays = np.logspace(-5, -1).tolist() + [0]
    clip_norms = np.logspace(-2, 3)
    dropouts = [0, .1, .25, .5]
    
    # Iterate over all combinations of hyperparameters
    for scale_factor in scale_factors:
        for weight_decay in weight_decays:
            for clip_norm in clip_norms:
                for dropout in dropouts:
                    # Initialize the model
                    model = GRUModel(input_size, hidden_size, output_size, dropout=dropout)
                    
                    # Train the model
                    train_model(model, task, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, clip_norm=clip_norm, weight_decay=weight_decay, scale_factor=scale_factor)
                    
                    # Save the model with a name that reflects the hyperparameters
                    model_name = f'model_sf{scale_factor}_wd{weight_decay}_cn{clip_norm}_do{dropout}.pth'
                    torch.save(model.state_dict(), os.path.join(exp_path, model_name))

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.GRU):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)
        
        
        
        
        
        
#######################ANALYSIS

def test_gru(model, task, batch_size=256):
    with torch.no_grad():
        inputs, targets, mask = task(batch_size)
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        targets = torch.tensor(targets, dtype=torch.float32).to(device)
        outputs, _ = model(inputs, targets)
        _, hs = model.sequence(inputs, targets)
    targets = targets.detach().numpy()
    outputs = outputs.detach().numpy()
    hs = hs.squeeze()
    trajectories = hs
    
    return inputs, targets, outputs, trajectories, hs

def gru_jacobian(model, h0):
    hidden_size = model.hidden_size
    
    # Input tensor at the origin
    #TODO: autonmatic input dim........
    x = torch.zeros(1, 1, 2)
    
    # Forward pass through the GRU
    out, hn = model.gru(x, h0)
    
    # Compute the Jacobian matrix
    jacobian = torch.zeros(hidden_size, hidden_size)
    
    for i in range(hidden_size):
        model.zero_grad()
        hn[0, 0, i].backward(retain_graph=True)
        jacobian[i] = h0.grad.view(-1)
        h0.grad.zero_()
    
    # Return the Jacobian matrix
    jacobian_matrix = jacobian.detach().numpy()
    return jacobian_matrix

def eigenspectrum_invman_gru(model, hs):
    hidden_size = model.hidden_size

    eigenspectrum = []
    for i,x in enumerate(hs):
        h0 = hs[i].clone().detach().unsqueeze(0).expand(1, -1, -1).requires_grad_(True)
        J = gru_jacobian(model,h0)
        if np.isfinite(J).all():
            eigenvalues, eigenvectors = np.linalg.eig(J)           
            eigenvalues = sorted(np.real(eigenvalues))
        else:
            eigenvalues = [np.nan]*hidden_size*2

        # plt.scatter([i]*hidden_size, eigenvalues, s=1, c='k', marker='o', alpha=0.5); 
        eigenspectrum.append(eigenvalues)
    return eigenspectrum

    
#OVERALL
def run_all_gru():
    df = pd.DataFrame(columns=['path', 'T', 'N'])
    T = 12.8
    dt = 0.1
    task = angularintegration_task(T, dt, sparsity='variable', random_angle_init=True)
    long_task = angularintegration_task_constant(T*10, dt, speed_range=[0,0], random_angle_init='equally_spaced')
    input_size = 1
    output_size = 2
    for N in [64,128,256]:
            
        hidden_size = N
        
        exp_path = f'C:\\Users\\abel_\\Documents\\Lab\\Projects\\topological_diversity/experiments/angular_integration_old/N{N}_T128_noisy/gru/'
        for exp_i in range(10):
            #load model
            model_path = exp_path+f'/model_{exp_i}.pth'
            model = GRUModel(input_size, hidden_size, output_size,dropout=0.)
            model.load_state_dict(torch.load(model_path))
            
            #run model on original task
            inputs, targets, outputs, trajectories, hs = test_gru(model, task, batch_size=256)
            mse, mse_normalized, db_mse_normalized = nmse(targets, outputs)
            print(exp_i,db_mse_normalized)
        
            #run model autonomously
            inputs, targets, outputs, trajectories, hs = test_gru(model, long_task, batch_size=256)
            
            #angular error
            min_error, mean_error, max_error, eps_min_int, eps_mean_int, eps_plus_int = angluar_error(targets, outputs)
            
            #fixed point
            fxd_pnt_output, fxd_pnt_thetas, stabilities = find_fixed_point(outputs)
            nfps = fxd_pnt_output.shape[0]
            fp_boa = boa(fxd_pnt_thetas)
            mean_fp_dist = mean_fp_distance(fxd_pnt_thetas)
            
            #VF uniform norm
            thetas = np.arctan2(outputs[:,:,0], outputs[:,:,1]);
            vf_infty = vf_norm_from_outtraj(thetas[:,126], thetas[:,127])
            
            #eigenspec
            eigenspectrum = eigenspectrum_invman_gru(model, hs[:,127,:])
        
            df = df.append({'path':model_path,
            'T': T, 'N': hidden_size, 'scale_factor': .5,  'dropout': 0., 'M': True, 'clip_gradient':1,
                        'trial': exp_i,
                        'mse': mse,
                        'mse_normalized':mse_normalized,
                        'nfps': nfps,
                        'stabilities':stabilities,
                        'boa':fp_boa,
                        'mean_fp_dist':mean_fp_dist,
                        'inv_man':trajectories[:,127,:],
                         'inv_man_output':outputs[:,127,:],
                         'vf_infty':vf_infty,
                         'eigenspectrum':eigenspectrum,
                          'min_error_0':min_error,
                           'mean_error_0':mean_error,
                            'max_error_0':max_error,
                            'eps_min_int_0':eps_min_int,
                            'eps_mean_int_0':eps_mean_int,
                            'eps_plus_int_0':eps_plus_int
                            }, ignore_index=True)
            
    return df




####################DOUBLE######################
# dt = 0.1
# batch_size = 4*256
# task = double_angularintegration_task(T, dt, sparsity='variable', random_angle_init=True)
# long_task = double_angularintegration_task(T*10, dt, speed_range=[0,0], random_angle_init='equally_spaced', constant_speed=True)
# input_size = 2
# output_size = 4
# for N in [64,128,256]:
#     hidden_size = N #  int(N/2)

#     exp_path = f'C:\\Users\\abel_\\Documents\\Lab\\Projects\\topological_diversity/experiments/double_angular/N{N}_T128/gru/'
#     for exp_i in range(1,10):
#         #load model
#         model_path = exp_path+f'/model_{exp_i}.pth'
#         model = GRUModel(input_size, hidden_size, output_size,dropout=0.)
#         model.load_state_dict(torch.load(model_path))

#         #run model on original task
#         inputs, targets, outputs, trajectories, hs = test_gru(model, task, batch_size=batch_size)
#         mse, mse_normalized, db_mse_normalized = nmse(targets, outputs)
#         print(db_mse_normalized)
        
#         inputs, targets, outputs, trajectories, hs = test_gru(model, long_task, batch_size=batch_size)
#         #if np.isnan(trajectories).any():
#         #    continue
#         #trajectories_flat = trajectories[:,:,:].reshape((-1,trajectories.shape[-1]));
#         #trajectories_pca = pca.fit_transform(trajectories_flat);
#         #trajectories_pca_time = trajectories_pca.reshape((batch_size,-1,10))
#         recurrences, recurrences_pca = find_periodic_orbits(trajectories, trajectories, limcyctol=1e-2, mindtol=1e-10)
#         fxd_pnts = np.array([recurrence[0] for recurrence in recurrences if len(recurrence)==1 or len(recurrence)==11]).reshape((-1,N));
#         #lcs =[recurrence for recurrence in recurrences if len(recurrence)!=1 and len(recurrence)>11];
#         if fxd_pnts.shape[0]==0:
#             continue
#         db = DBSCAN(eps=epsilon, min_samples=1).fit(fxd_pnts);
#         unique_indices = np.unique(db.labels_, return_index=True)[1]
#         fps = fxd_pnts[unique_indices]
#         fps_out = model.fc(torch.tensor(fps[:,:hidden_size]))
#         fps_out = fps_out.detach().numpy() 
#         #fps_out = np.dot(fps, wo)
#         nfps = unique_indices.shape[0]
#         #stabilities = stabilities_fps(wrec, brec, fps, rnn_ode_jacobian)
#         fxd_pnt_thetas1 = np.arctan2(fps_out[...,1], fps_out[...,0]);
#         fxd_pnt_thetas2 = np.arctan2(fps_out[...,3], fps_out[...,2]);
#         print(exp_i, nfps, db_mse_normalized)
        
#         mean_fp_dist = double_mean_fp_distance(fxd_pnt_thetas1, fxd_pnt_thetas2)
#         fp_boa = double_boa(fxd_pnt_thetas1, fxd_pnt_thetas2)
        
#         thetas1 = np.arctan2(outputs[:,:,1], outputs[:,:,0]);
#         thetas2 = np.arctan2(outputs[:,:,3], outputs[:,:,2]);
#         vf_infty1 = vf_norm_from_outtraj(thetas1[:,126], thetas1[:,127])
#         vf_infty2 = vf_norm_from_outtraj(thetas1[:,126], thetas1[:,127])
#         vf_infty = vf_infty1+vf_infty2
#         eigenspectrum = eigenspectrum_invman_gru(model, hs[:,127,:])

#         min_error, mean_error, max_error, eps_min_int, eps_mean_int, eps_plus_int = angular_double_error(targets, outputs)
#         df_gru = df_gru.append({'path':model_path, 'S':'lstm',
# 'T': T, 'N': N, 'scale_factor': .5,  'dropout': 0., 'M': True, 'clip_gradient':1,
#             'trial': exp_i,
#             'mse': mse,
#             'mse_normalized':mse_normalized,
#             'nfps': nfps,
#             'fps':fps,
#             'fps_out':fps_out,
#             #'stabilities':stabilities,
#             'boa':fp_boa,
#             'mean_fp_dist':mean_fp_dist,
#             'inv_man':trajectories[:,0,:],
#              'inv_man_output':outputs[:,0,:],
#              'vf_infty':vf_infty,
#              'eigenspectrum':eigenspectrum,
#               'min_error_0':min_error,
#                'mean_error_0':mean_error,
#                 'max_error_0':max_error,
#                 'eps_min_int_0':eps_min_int,
#                 'eps_mean_int_0':eps_mean_int,
#                 'eps_plus_int_0':eps_plus_int
#                 }, ignore_index=True)