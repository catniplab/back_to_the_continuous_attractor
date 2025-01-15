# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:53:54 2024

@author: 
"""

import os, sys

from tasks import angularintegration_task, angularintegration_task_constant, double_angularintegration_task

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt; from matplotlib.ticker import MaxNLocator

from analysis_functions import calculate_lyapunov_spectrum, participation_ratio, identify_limit_cycle, find_periodic_orbits, find_analytic_fixed_points, powerset


# from analysis_functions import db
def decibel(x):
    return 10*np.log10(x)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

tanh = nn.Tanh()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
        
class NoisyLSTM(nn.LSTM):
    def __init__(self, *args, noise_std=0.1, **kwargs):
        super(NoisyLSTM, self).__init__(*args, **kwargs)
        self.noise_std = noise_std
    
    def forward(self, input, hx=None):
        if hx is None:
            num_layers = self.num_layers * 2 if self.bidirectional else self.num_layers
            h_0 = torch.zeros(num_layers, input.size(1), self.hidden_size, device=input.device, dtype=input.dtype)
            c_0 = torch.zeros(num_layers, input.size(1), self.hidden_size, device=input.device, dtype=input.dtype)
            hx = (h_0, c_0)
        
        h_0, c_0 = hx
        h_0 = h_0 + torch.randn_like(h_0) * self.noise_std
        c_0 = c_0 + torch.randn_like(c_0) * self.noise_std

        output, (h_n, c_n) = super(NoisyLSTM, self).forward(input, (h_0, c_0))

        # Inject noise at each time step
        h_n = h_n + torch.randn_like(h_n) * self.noise_std
        c_n = c_n + torch.randn_like(c_n) * self.noise_std

        return output, (h_n, c_n)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.5, init_weight_radius_scaling=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #self.output_to_hidden = nn.Linear(output_size, hidden_size)
        self.output_to_hidden = nn.Parameter(torch.Tensor(output_size, hidden_size))
        #self.output_to_cell = nn.Linear(output_size, hidden_size)
        self.output_to_cell = nn.Parameter(torch.Tensor(output_size, hidden_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights with a smaller standard deviation for LSTM weights
        self.init_weight_radius_scaling = init_weight_radius_scaling
        self._initialize_weights()
    
    def _initialize_weights(self):
        stdv = self.init_weight_radius_scaling / np.sqrt(self.hidden_size)
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:  # Only initialize recurrent weights
                nn.init.uniform_(param, -stdv, stdv)

    def forward(self, x, target):
        
        batch_size = x.shape[0]
        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        h_init_torch = nn.Parameter(torch.Tensor(self.num_layers, batch_size, self.hidden_size)).to(x.device)
        h_init_torch.requires_grad = False
        h_init_torch = target[:,0,:].matmul(self.output_to_hidden)
        h_init_torch = tanh(h_init_torch)
            
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_init_torch = nn.Parameter(torch.Tensor(self.num_layers, batch_size, self.hidden_size)).to(x.device)
        c_init_torch.requires_grad = False
        c_init_torch = target[:,0,:].matmul(self.output_to_cell)
        c_init_torch = tanh(h_init_torch)
        
        h0 = h_init_torch.unsqueeze(0).expand(self.num_layers, -1, -1)  # (num_layers, batch_size, hidden_size)
        c0 = c_init_torch.unsqueeze(0).expand(self.num_layers, -1, -1)  # (num_layers, batch_size, hidden_size)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out)
        return out, hn, cn
    
    def forward_nooth(self, x, h_init_torch, c_init_torch):
        
        h0 = h_init_torch.unsqueeze(0).expand(self.num_layers, -1, -1)  # (num_layers, batch_size, hidden_size)
        c0 = c_init_torch.unsqueeze(0).expand(self.num_layers, -1, -1)  # (num_layers, batch_size, hidden_size)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out)
        return out, hn, cn
    
    def sequence(self, x, target):
            batch_size = x.shape[0]
            
            # Initialize h_init_torch and c_init_torch
            h_init_torch = target[:, 0, :].matmul(self.output_to_hidden)
            h_init_torch = tanh(h_init_torch)
            
            c_init_torch = target[:, 0, :].matmul(self.output_to_cell)
            c_init_torch = tanh(c_init_torch)
            
            # Ensure h0 and c0 have the correct dimensions
            h0 = h_init_torch.unsqueeze(0).expand(self.num_layers, batch_size, self.hidden_size).contiguous()
            c0 = c_init_torch.unsqueeze(0).expand(self.num_layers, batch_size, self.hidden_size).contiguous()
            
            # Initialize lists to store all hidden and cell states
            all_out = []
            all_hidden_states = []
            all_cell_states = []
        
            # Get the sequence length
            seq_len = x.size(1)  # Assuming x is of shape (batch_size, seq_len, input_size)
        
            # Iterate through the sequence
            for t in range(seq_len):
                # Get the input at time step t
                xt = x[:, t, :].unsqueeze(1)
                
                # Apply LSTM cell
                out, (h0, c0) = self.lstm(xt, (h0, c0))
                
                # Store the hidden and cell states
                all_out.append(out)
                all_hidden_states.append(h0[-1].unsqueeze(1))
                all_cell_states.append(c0[-1].unsqueeze(1))
    
            # Stack the states to form tensors in the correct shape (batch_size, T, N)
            all_out = torch.cat(all_out, dim=1)
            all_hidden_states = torch.cat(all_hidden_states, dim=1)
            all_cell_states = torch.cat(all_cell_states, dim=1)

            return all_out, all_hidden_states, all_cell_states
        
def extract_lstm_parameters(model):
    hidden_size = model.hidden_size
    params = {}

    # Extract weights and biases for input, forget, cell, and output gates
    params['W_i'] = model.weight_ih_l0[:hidden_size, :].detach().numpy()
    params['W_f'] = model.weight_ih_l0[hidden_size:2*hidden_size, :].detach().numpy()
    params['W_c'] = model.weight_ih_l0[2*hidden_size:3*hidden_size, :].detach().numpy()
    params['W_o'] = model.weight_ih_l0[3*hidden_size:, :].detach().numpy()

    params['U_i'] = model.weight_hh_l0[:hidden_size, :].detach().numpy()
    params['U_f'] = model.weight_hh_l0[hidden_size:2*hidden_size, :].detach().numpy()
    params['U_c'] = model.weight_hh_l0[2*hidden_size:3*hidden_size, :].detach().numpy()
    params['U_o'] = model.weight_hh_l0[3*hidden_size:, :].detach().numpy()

    params['b_i'] = model.bias_ih_l0[:hidden_size].detach().numpy() + model.bias_hh_l0[:hidden_size].detach().numpy()
    params['b_f'] = model.bias_ih_l0[hidden_size:2*hidden_size].detach().numpy() + model.bias_hh_l0[hidden_size:2*hidden_size].detach().numpy()
    params['b_c'] = model.bias_ih_l0[2*hidden_size:3*hidden_size].detach().numpy() + model.bias_hh_l0[2*hidden_size:3*hidden_size].detach().numpy()
    params['b_o'] = model.bias_ih_l0[3*hidden_size:].detach().numpy() + model.bias_hh_l0[3*hidden_size:].detach().numpy()

    return params

def test_model(model, task, batch_size):
    with torch.no_grad():
        inputs, targets, mask = task(batch_size)
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        targets = torch.tensor(targets, dtype=torch.float32).to(device)
        outputs, _, _ = model(inputs, targets);
        _, hs, cs = model.sequence(inputs, targets);
    targets = targets.detach().numpy()
    outputs = outputs.detach().numpy(); hs = hs.detach().numpy(); cs = cs.detach().numpy(); hs=hs.squeeze(); cs=cs.squeeze();
    trajectories = np.concatenate((hs,cs),axis=-1); trajectories = np.concatenate((hs,cs),axis=-1); 
    from_t=0; to_t=None; target_power = np.mean(targets[:,from_t:to_t,:]**2)
    mse = np.mean((targets[:,from_t:to_t,:] - outputs[:,from_t:to_t,:])**2)
    mse_normalized = mse/target_power
    return decibel(mse_normalized), outputs, trajectories

def train_model(model, task, num_epochs=100, batch_size=32, learning_rate=0.001, 
                clip_norm=0., output_noise_level=0.01, weight_decay=0.01, scale_factor=1.):
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
                            {'params': model.lstm.parameters(), 'weight_decay': weight_decay},
                            {'params': model.fc.parameters(), 'weight_decay': 0.0},
                            {'params': model.output_to_hidden, 'weight_decay': 0.0},
                            {'params': model.output_to_cell, 'weight_decay': 0.0}
                        ], lr=learning_rate)
    
    model.to(device)
    criterion.to(device)
    losses = []
    for epoch in range(num_epochs):
        inputs, targets, mask = task(batch_size)
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        targets = torch.tensor(targets, dtype=torch.float32).to(device)
        output_noise = torch.randn(batch_size, inputs.shape[1], targets.shape[-1]).to(device)*output_noise_level
        targets += output_noise
        mask = torch.tensor(mask, dtype=torch.float32).to(device)
        
        # Save model and optimizer state
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()

        outputs, _, _ = model(inputs, targets)
        
        # Check for inf values in outputs
        if torch.isinf(outputs).any():
            print(f'Inf detected in outputs at epoch {epoch + 1}. Scaling down weights.')
            with torch.no_grad():
                for param in model.parameters():
                    param *= scale_factor
        
        loss = criterion(outputs * mask, targets * mask)
        
        # Check for NaNs in loss
        if torch.isnan(loss):
            return False
            # if epoch == 0:
            #     print(f'NaN detected in loss at epoch {epoch + 1}. Reinitializing model parameters.')
            #     model.apply(init_weights)
            #     optimizer = optim.Adam([
            #                 {'params': model.lstm.parameters(), 'weight_decay': weight_decay},
            #                 {'params': model.fc.parameters(), 'weight_decay': 0.0},
            #                 {'params': model.output_to_hidden, 'weight_decay': 0.0},
            #                 {'params': model.output_to_cell, 'weight_decay': 0.0}
            #             ], lr=learning_rate)
            # else:
            #     print(f'NaN detected in loss at epoch {epoch + 1}. Rolling back to previous state.')
            #     model.load_state_dict(model_state_dict)
            #     optimizer.load_state_dict(optimizer_state_dict)
            # continue
        
        optimizer.zero_grad()
        loss.backward()
        if clip_norm>0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        
        losses.append(loss.item())  # Save the loss value

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        
    losses_array = np.array(losses)
    return losses_array
            
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)

def train_n(n, task, input_size = 1, hidden_size = 64,  output_size = 2,
            num_epochs=5000, batch_size=64, learning_rate=0.001, clip_norm=1, dropout=0.,
            exp_path='C:/Users/abel_/Documents/Lab/projects/topological_diversity/experiments/angular_integration_old/N128_T128_noisy/lstm/'):
    makedirs(exp_path)
    i=0
    while i<n:
        model = LSTMModel(input_size, hidden_size, output_size, dropout=dropout)
        losses = train_model(model, task, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, clip_norm=clip_norm);
        if not losses.any():
            continue
        torch.save(model.state_dict(), exp_path+f'/model_{i}.pth')
        i+=1

def grid_search(task, input_size = 1, hidden_size = 64, output_size = 2,
            num_epochs=5000, batch_size=64, learning_rate=0.001, dropout=0.,
            exp_path=''):
    
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    scale_factors = [0.8,.9,0.99,1.]
    weight_decays = np.logspace(-5, -1).tolist() + [0]
    clip_norms = np.logspace(-2,3)
    dropouts = [0,.1,.25,.5]
    
    # Iterate over all combinations of hyperparameters
    for scale_factor in scale_factors:
        for weight_decay in weight_decays:
            for clip_norm in clip_norms:
                for dropout in dropouts:
                    # Initialize the model
                    model = LSTMModel(input_size, hidden_size, output_size, dropout=dropout)
                    
                    # Train the model
                    train_model(model, task, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, clip_norm=clip_norm, weight_decay=weight_decay, scale_factor=scale_factor)
                    
                    # Save the model with a name that reflects the hyperparameters
                    model_name = f'model_sf{scale_factor}_wd{weight_decay}_cn{clip_norm}_do{dropout}.pth'
                    torch.save(model.state_dict(), os.path.join(exp_path, model_name))

#     fig = plt.figure(figsize=(3, 3));
#     ax = fig.add_subplot(111)
#     xs = np.linspace(0,2*np.pi,100)
#     ax.plot(np.cos(xs), np.sin(xs))
#     for i in range(batch_size):
#         ax.plot(outputs[i,:,0], outputs[i,:,1]);
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     ax.yaxis.set_major_locator(MaxNLocator(integer=True))






# fig, ax = plt.subplots(figsize=(2,3))
# all_data = []
# largest = []; second=[]; rest=[]
# for exp_i in range(len(df)):
#     eigenspectrum = np.array(df['eigenspectrum'][exp_i])-1
#     largest.extend(eigenspectrum[:, -1])
#     second.extend(eigenspectrum[:, -2])
#     rest.extend(eigenspectrum[:, :-2].flatten())
# largest = np.array(largest)
# second = np.array(second)
# rest = np.array(rest)

# all_data = [largest[~np.isnan(largest)],second[~np.isnan(second)],rest[~np.isnan(rest)]]
# labels = ['largest', 'second largest', 'rest']
# colors = ['b', 'purple', 'k']

# # Plot each violin plot with a slight offset on the x-axis
# positions = [1, 1., 1.]  # Slightly offset positions for each violin plot
# violin_parts = ax.violinplot(all_data, positions=positions, showmeans=False, showmedians=True)

# # Set colors and labels for each violin
# for i, pc in enumerate(violin_parts['bodies']):
#     pc.set_facecolor(colors[i])
#     pc.set_edgecolor(colors[i])
#     pc.set_alpha(0.5)

# # Create custom legend
# for i in range(len(labels)):
#     ax.plot([], [], color=colors[i], label=labels[i], alpha=0.5)

# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# # Set x-ticks to a single value
# ax.set_xticks([])
# ax.set_ylabel("eigenvalues")
# plt.savefig(exp_path+'/all_eigenvalue_spectra.pdf',bbox_inches="tight", pad_inches=0.0)


######################################PLOT NFPS vs MAE
# fig, ax = plt.subplots(1, 1, figsize=(5, 3))
# N_color_dict = {64: 'b', 128: 'g', 256: 'orange'}
# np.random.seed(12111)
# for N in [64, 128, 256]:
#     df_N = df[df['N'] == int(N / 2)]
#     error_T1 = np.array(df_N['mean_error_0'].tolist())[:, 127]
#     jittered_nfps = add_jitter(np.array(df_N['nfps']))
#     ax.plot(jittered_nfps, error_T1, 'X', color=N_color_dict[N],alpha=.7)
#     error_10T1 = np.array(df_N['mean_error_0'].tolist())[:, -1]
#     ax.plot(jittered_nfps, error_10T1, 'X', color=N_color_dict[N], fillstyle='none',alpha=.7)

#     ax.plot(jittered_nfps, df_N['mean_fp_dist'], '.', color='magenta')

# ax.set_ylim([0, 1.25])

# lines = [Line2D([0], [0], color=c, linewidth=2, linestyle='-') for c in ['b', 'g', 'orange']]
# first_legend = plt.legend(lines, N_color_dict.keys(), title='Network Size', loc='upper right')
# ax.add_artist(first_legend)

# plt.xlabel("number of fixed points")
# plt.ylabel("angular error (rad)")
# plt.savefig('C:\\Users\\abel_\\Documents\\Lab\\Projects\\topological_diversity/experiments/angular_integration_old/lstm_npfs_vs_mae.pdf');


######################################PLOT VFNorm vs MAE
# fig, ax = plt.subplots(1, 1, figsize=(5, 3));
# N_color_dict = {64:'b', 128:'g', 256:'orange'}
# for N in [64,128,256]:
#     df_N = df[df['N'] == int(N/2)]
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
# plt.savefig('C:\\Users\\abel_\\Documents\\Lab\\Projects\\topological_diversity/experiments/angular_integration_old/lstm_vfnorm_vs_mae.pdf');




#######################ANALYSIS


def test_lstm(model, task, batch_size=256):
    
    with torch.no_grad():
        inputs, targets, mask = task(batch_size)
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        outputs, _, _ = model(inputs, targets);
        _, hs, cs = model.sequence(inputs, targets);
    targets = targets.detach().numpy()
    outputs = outputs.detach().numpy(); 
    hs_np = hs.detach().numpy(); cs_np = cs.detach().numpy(); hs=hs.squeeze(); cs=cs.squeeze();
    trajectories = np.concatenate((hs_np,cs_np),axis=-1);
    
    return inputs, targets, outputs, trajectories, hs, cs

def nmse(targets, outputs, from_t=0, to_t=None):
    target_power = np.mean(targets[:,from_t:to_t,:]**2)
    mse = np.mean((targets[:,from_t:to_t,:] - outputs[:,from_t:to_t,:])**2)
    mse_normalized = mse/target_power
    return mse, mse_normalized, decibel(mse_normalized)


def angluar_error(target, output):
    output_angle = np.arctan2(output[:,:,1], output[:,:,0]);
    target_angle = np.arctan2(target[:,:,1], target[:,:,0]);
    angle_error = np.abs(output_angle - target_angle)
    angle_error[np.where(angle_error>np.pi)] = 2*np.pi-angle_error[np.where(angle_error>np.pi)]
    
    mean_error = np.mean(angle_error,axis=0)
    max_error = np.max(angle_error,axis=0)
    min_error = np.min(angle_error,axis=0)

    eps_mean_int = np.cumsum(mean_error) / np.arange(1,1+mean_error.shape[0])
    eps_plus_int = np.cumsum(max_error) / np.arange(1,1+mean_error.shape[0])
    eps_min_int = np.cumsum(min_error) / np.arange(1,1+mean_error.shape[0])
    
    return min_error, mean_error, max_error, eps_min_int, eps_mean_int, eps_plus_int
 



def find_fixed_point(outputs):
    
    trajectories_proj2 = outputs
    thetas = np.arctan2(trajectories_proj2[:,:,0], trajectories_proj2[:,:,1]);

    thetas_init = thetas[:,0] #np.arange(-np.pi, np.pi, np.pi/batch_size*2);
    idx = np.argsort(thetas_init)
    thetas_init = thetas_init[idx]    
    
    idx = np.argsort(thetas_init)
    thetas_init = thetas_init[idx]; 
    thetas_sorted = thetas[idx]
    outputs_sorted = outputs[idx]
    theta_unwrapped = np.unwrap(thetas_sorted, period=2*np.pi);
    theta_unwrapped = np.roll(theta_unwrapped, -1, axis=0);
    arr = np.sign(theta_unwrapped[:,-1]-theta_unwrapped[:,0]);
    idx=np.array([i for i, t in enumerate(zip(arr, arr[1:])) if t[0] != t[1]]).astype(int)  
    stabilities=-arr[idx].astype(int)

    fxd_pnt_output = outputs_sorted[:,-1,:][idx]
    fxd_pnt_thetas = thetas_init[idx]
    
    if idx.shape[0]%2==1:
        fxd_pnt_thetas=np.hstack([fxd_pnt_thetas,0])
        fxd_pnt_output=np.hstack([fxd_pnt_output,[1,0]])
        stabilities=np.append(stabilities, -np.sum(stabilities))

    return fxd_pnt_output, fxd_pnt_thetas, stabilities


def boa(fxd_pnt_thetas):
    nfps = fxd_pnt_thetas.shape[0]
    boas = []
    for i in range(0,nfps,2):
        boa = (fxd_pnt_thetas[i+1]-fxd_pnt_thetas[i-1])% (2*np.pi)
        boas.append(boa/np.pi/2)
    if nfps>2:
        perf = -np.sum([boa*np.log(boa) for boa in boas])
    else:
        perf=0
    return perf

def mean_fp_distance(fxd_pnt_thetas):
    if fxd_pnt_thetas.shape[0]==0:
        return np.inf
    pairwise_distances = np.diff(fxd_pnt_thetas)
    pairwise_distances = np.mod(pairwise_distances, 2*np.pi)
    return np.mean(pairwise_distances)


def vf_norm_from_outtraj(angle_t, angle_tplus1):
    return np.max(np.linalg.norm(angle_tplus1-angle_t,axis=-1))


def lstm_jacobian(model, h0, c0):
    hidden_size = model.hidden_size
    
    # Input tensor at the origin
    x = torch.zeros(1, 1, 2)
    
    # Forward pass through the LSTM
    out, (hn, cn) = model.lstm(x, (h0, c0))
    
    # Jointly concatenate hidden and cell states
    joint_state = torch.cat((hn.view(-1), cn.view(-1)))
    
    # Compute the Jacobian matrix
    jacobian = torch.zeros(2*hidden_size, 2*hidden_size)
    
    for i in range(2*hidden_size):
        model.zero_grad()
        joint_state[i].backward(retain_graph=True)
        jacobian[i] = torch.cat((h0.grad.view(-1), c0.grad.view(-1)))
        h0.grad.zero_()
        c0.grad.zero_()
    
    # Return the Jacobian matrix
    jacobian_matrix = jacobian.detach().numpy()
    return jacobian_matrix

def eigenspectrum_invman_lstm(model, hs, cs):
    hidden_size = model.hidden_size

    eigenspectrum = []
    for i,x in enumerate(hs):
        h0 = hs[i].clone().detach().unsqueeze(0).expand(1, -1, -1).requires_grad_(True)
        c0 = cs[i].clone().detach().unsqueeze(0).expand(1, -1, -1).requires_grad_(True)

        # h0 = torch.tensor(h0, dtype=torch.float32, requires_grad=True).unsqueeze(0).expand(1, -1, -1)
        # c0 = torch.tensor(c0, dtype=torch.float32, requires_grad=True).unsqueeze(0).expand(1, -1, -1)
        
        J = lstm_jacobian(model,h0,c0)
        if np.isfinite(J).all():
            eigenvalues, eigenvectors = np.linalg.eig(J)           
            eigenvalues = sorted(np.real(eigenvalues))
        else:
            eigenvalues = [np.nan]*hidden_size*2

        # plt.scatter([i]*hidden_size, eigenvalues, s=1, c='k', marker='o', alpha=0.5); 
        eigenspectrum.append(eigenvalues)
    return eigenspectrum
    # plt.plot(max_eigv, 'b', label='1st')
    # plt.plot(max_eigv_2nd, 'purple', label='2nd')
    # plt.xlabel(r'$\theta$')
    # plt.ylabel('eigenvalue spectrum')
    # plt.xticks([0,len(csx)], [0,r'$2\pi$'])
    # #plt.ylim([-1.5,0.2])
    # plt.legend(loc='lower right')
    # plt.hlines(0, 0,len(csx), 'r', linestyles='dotted');
    # plt.savefig(exp_path+f'/eigenvalue_spectrum_{exp_i}.pdf')
    

#################PLOT
def plot_outputs_fps(outputs, fxd_pnt_output, stabilities,
        exp_path='', exp_i=None):
    
    fig = plt.figure(figsize=(3, 3));
    stab_colors = np.array(['k', 'red', 'g'])
    ax = fig.add_subplot(111)
    xs = np.linspace(0,2*np.pi,100)
    ax.plot(np.cos(xs), np.sin(xs), 'grey', linewidth=10)
    #for i in range(batch_size):
    #ax.plot(outputs[:,0,0], outputs[:,0,1]);
    ax.plot(outputs[:,-1,0], outputs[:,-1,1], 'k');
    #ax.plot(fxd_pnt_output[:,0], fxd_pnt_output[:,1], '.')
    ax.scatter(fxd_pnt_output[:,0], fxd_pnt_output[:,1], marker='o',
               c=stab_colors[stabilities], alpha=1, zorder=101)
    ax.set_xlim([-1.4,1.4])
    ax.set_ylim([-1.4,1.4])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True));
    plt.savefig(exp_path+f'/inv_man_fps_{exp_i}.pdf');
    
    
    
    
#OVERALL
def run_all_lstm():
    df = pd.DataFrame(columns=['path', 'T', 'N'])
    T = 12.8
    dt = 0.1
    task = angularintegration_task(T, dt, sparsity='variable', random_angle_init=True)
    long_task = angularintegration_task_constant(T*10, dt, speed_range=[0,0], random_angle_init='equally_spaced')
    input_size = 1
    output_size = 2
    for N in [64,128,256]:
            
        hidden_size = int(N/2)
        
        exp_path = f'C:\\Users\\abel_\\Documents\\Lab\\Projects\\topological_diversity/experiments/angular_integration_old/N{N}_T128_noisy/lstm/'
        for exp_i in range(10):
            #load model
            model_path = exp_path+f'/model_{exp_i}.pth'
            model = LSTMModel(input_size, hidden_size, output_size,dropout=0.)
            model.load_state_dict(torch.load(model_path))
            
            #run model on original task
            inputs, targets, outputs, trajectories, hs, cs = test_lstm(model, task, batch_size=256)
            mse, mse_normalized, db_mse_normalized = nmse(targets, outputs)
            print(db_mse_normalized)
        
            #run model autonomously
            inputs, targets, outputs, trajectories, hs, cs = test_lstm(model, long_task, batch_size=256)
            
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
            eigenspectrum = eigenspectrum_invman_lstm(model, hs[:,127,:], cs[:,127,:])
        
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



   
class LSTM_noforget(nn.Module):
    def __init__(self, dims, readout_nonlinearity='id', w_init=None, u_init=None, wo_init=None, bias_init=None, dropout=0.):
        super(LSTM_noforget, self).__init__()
        self.dims = dims
        input_size, hidden_size, output_size = dims
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.readout_nonlinearity = readout_nonlinearity
        
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 3))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 3))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 3))
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        
        self.drop = nn.Dropout(p=dropout)
        
        # Initialize parameters
        with torch.no_grad():
            k = np.sqrt(1/hidden_size)
            if w_init is None:
                torch.nn.init.uniform_(self.W, a=-k, b=k)
                # torch.nn.init.normal_(self.W, std=1/hidden_size)
            else:
                if type(w_init) == np.ndarray:
                    w_init = torch.from_numpy(w_init)
                self.W.copy_(w_init)
            if u_init is None:
               torch.nn.init.uniform_(self.U, a=-k, b=k)
            else:
                if type(u_init) == np.ndarray:
                    u_init = torch.from_numpy(u_init)
            if bias_init is None:
                # self.bias.normal_(std=1 / hidden_size)
                torch.nn.init.uniform_(self.bias, a=-k, b=k)
            else:
                if type(bias_init) == np.ndarray:
                    bias_init = torch.from_numpy(bias_init)
                self.bias.copy_(bias_init)
            
            if wo_init is None:
                # self.wo.normal_(std=1 / hidden_size)
                torch.nn.init.uniform_(self.wo, a=-k, b=k)

            else:
                if type(wo_init) == np.ndarray:
                    wo_init = torch.from_numpy(wo_init)
                self.wo.copy_(wo_init)

        
        # Readout nonlinearity
        if readout_nonlinearity == 'tanh':
            self.readout_nonlinearity = torch.tanh
        elif readout_nonlinearity == 'logistic':
            # Note that the range is [0, 1]. otherwise, 'logistic' is a scaled and shifted tanh
            self.readout_nonlinearity = lambda x: 1. / (1. + torch.exp(-x))
        elif readout_nonlinearity == 'id':
            self.readout_nonlinearity = lambda x: x
        elif type(readout_nonlinearity) == str:
            raise NotImplementedError("readout_nonlinearity not yet implemented.")
        else:
            self.readout_nonlinearity = readout_nonlinearity
                
         
    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_size, sequence_length, _ = x.size()
        hidden_seq = []
        out_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(x.device), 
                        torch.zeros(batch_size, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(sequence_length):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.tanh(gates[:, HS:HS*2]),
                torch.sigmoid(gates[:, HS*2:]), # output
            )
            c_t = i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            
            hidden_seq.append(h_t.unsqueeze(0))
            
            # print(i_t, g_t, o_t, h_t)
            
            # out_t = self.readout_nonlinearity(h_t.matmul(self.wo))
            # out_seq.append(o_t.unsqueeze(0))
            
            
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # hidden_seq = self.drop(hidden_seq)
        out_seq = self.readout_nonlinearity(hidden_seq.matmul(self.wo))

        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        out_seq = out_seq.transpose(0, 1).contiguous()

        # print(hidden_seq)
        return out_seq, hidden_seq, (h_t, c_t, 0)
    
    
class LSTM_noforget2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_noforget2, self).__init__()
        self.hidden_size = hidden_size
        
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        bs, sequence_length, _ = x.size()
        h_t = torch.zeros(1, self.hidden_size)
        hidden_seq = []
        for t in range(sequence_length):
            combined = torch.cat((x[:,t,:], h_t), dim=1)
            
            i_t = torch.sigmoid(self.input_gate(combined))
            g_t = torch.tanh(self.cell_gate(combined))
            o_t = torch.sigmoid(self.output_gate(combined))
            c_t = i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            
            output = self.output_layer(h_t)
            hidden_seq.append(h_t.unsqueeze(0))
        
        return output, hidden
    
    
def train_lstm(net, task, n_epochs, batch_size=32, learning_rate=1e-2, clip_gradient=None, cuda=False,
          loss_function='mse', init_states=None, final_loss=True, last_mses=None,
          optimizer='sgd', momentum=0, weight_decay=.0, adam_betas=(0.9, 0.999), adam_eps=1e-8, #optimizers 
          scheduler=None, scheduler_step_size=100, scheduler_gamma=0.3, 
          verbose=True, record_step=1):
    
    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net.to(device=device)
    
    # Optimizer
    optimizer = get_optimizer(net, optimizer, learning_rate, weight_decay, momentum, adam_betas, adam_eps)
    scheduler = get_scheduler(net, optimizer, scheduler, scheduler_step_size, scheduler_gamma, n_epochs)
    loss_function = get_loss_function(net, loss_function)
    
    # Save initial weights
    # w_init = net.W.cpu().detach().numpy().copy()
    # u_init = net.U.cpu().detach().numpy().copy()
    # bias_init = net.bias.cpu().detach().numpy().copy()
    # wo_init = net.wo.cpu().detach().numpy().copy()
    # weights_init = [w_init, u_init, bias_init, wo_init]
    weights_init = []
    for name, param in net.named_parameters():
        weights_init.append(param.cpu().detach().numpy().copy())
    
    
    # Record
    losses = np.zeros((n_epochs), dtype=np.float32)
    gradient_norm_sqs = np.zeros((n_epochs), dtype=np.float32)
    epochs = np.zeros((n_epochs))
    
    dim_rec = net.hidden_size
    dim_in = net.input_size
    dim_out = net.output_size
    n_rec_epochs = n_epochs // record_step
    
    #TODO: write in general form (loop over params)
    rec_epochs = np.zeros((n_rec_epochs))
    # ws = np.zeros((n_rec_epochs, dim_in, dim_rec*3), dtype=np.float32)
    # us = np.zeros((n_rec_epochs, dim_rec, dim_rec*3), dtype=np.float32)
    # biases = np.zeros((n_rec_epochs, dim_rec*3), dtype=np.float32)
    # wos = np.zeros((n_rec_epochs, dim_rec, dim_out), dtype=np.float32)
    
    time0 = time.time()
    if verbose:
        print("Training...")
    for i in range(n_epochs):
        # Save weights (before update)
        # if i % record_step == 0:
        #     k = i // record_step
        #     rec_epochs[k] = i
        #     ws[k] = net.W.cpu().detach().numpy()
        #     us[k] = net.U.cpu().detach().numpy()
        #     biases[k] = net.bias.cpu().detach().numpy()
        #     wos[k] = net.wo.cpu().detach().numpy()
        
        # Generate batch
        _input, _target, _mask = task(batch_size)
        # Convert training data to pytorch tensors
        _input = torch.from_numpy(_input)
        _target = torch.from_numpy(_target)
        _mask = torch.from_numpy(_mask)
        # Allocate
        input = _input.to(device=device).float() 
        target = _target.to(device=device).float() 
        # mask = _mask.to(device=device).float() 
        
        optimizer.zero_grad()
        try:        #TODO: write general forward pass
            output, _, _ = net(input, init_states=init_states)
        except:
            output = net(input)
        if final_loss:
            if not last_mses:
                last_mses = output.shape[1]
            fin_int = np.random.randint(1,last_mses,size=batch_size)
            loss = loss_function(output[:,-fin_int,:], target[:,-fin_int,:])
        else:
            loss = loss_function(output, target)
              
        # Gradient descent
        loss.backward()
        if clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
        gradient_norm_sq = sum([(p.grad ** 2).sum() for p in net.parameters() if p.requires_grad])
        
        # Update weights
        optimizer.step()
        
        scheduler.step()
        
        #To prevent memory leaks
        loss.detach()
        output.detach()
        
        # Save
        epochs[i] = i
        losses[i] = loss.item()
        gradient_norm_sqs[i] = gradient_norm_sq
        
        if verbose:
            print("epoch %d / %d:  loss=%.6f \n" % (i+1, n_epochs, np.mean(losses[i])))
            
    if verbose:
        print("\nDone. Training took %.1f sec." % (time.time() - time0))
    
    # Obtain gradient norm
    gradient_norms = np.sqrt(gradient_norm_sqs)
    
    # Weights throughout training: 
    weights_train = {}
    # weights_train["w"] = ws
    # weights_train["u"] = us
    # weights_train["wo"] = wos
    # weights_last =  [net.W, net.U, net.bias, net.wo]
    weights_last = []
    for name, param in net.named_parameters():
        weights_last.append(param.cpu().detach().numpy().copy())
    
    # res = [losses, gradient_norms, weights_last, epochs]
    # return res
    res = [losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs]
    return res
        


def run_lstm(net, task, batch_size=32, return_dynamics=False, init_states=None):
    loss_fn = nn.MSELoss()

    # Generate batch
    input, target, mask = task(batch_size)
    # Convert training data to pytorch tensors
    input = torch.from_numpy(input).float() 
    target = torch.from_numpy(target).float() 
    mask = torch.from_numpy(mask).float() 
    with torch.no_grad():
        # Run dynamics
        output, hidden_seq, _ = net(input, init_states=init_states)

        loss = loss_fn(output, target)
    res = [input, target, mask, output, loss]
    if return_dynamics:
        res.append(hidden_seq)
    res = [r.numpy() for r in res]
    return res

# df = run_all_lstm()