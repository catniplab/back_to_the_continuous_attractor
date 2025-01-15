# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:44:55 2024
"""

import os, sys
import glob
import pickle
import yaml
from pathlib import Path
current_dir = os.path.dirname(os.path.realpath('__file__'))

        
from tqdm import tqdm
import numpy as np
import torch
from scipy.spatial.distance import cdist
import scipy
import pandas as pd
from ripser import ripser
from persim import plot_diagrams
#import skdim

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

from models import RNN 
from tasks import angularintegration_task, angularintegration_task_constant, center_out_reaching_task, double_angularintegration_task
from odes import relu_ode, tanh_ode, recttanh_ode, relu_jacobian, tanh_jacobian, recttanh_jacobian_point, get_rnn_ode
from analysis_functions import decibel
from load_network import get_params_exp, load_net_from_weights, load_net_path, get_weights_from_net, get_tr_par
from simulate_network import simulate_rnn, simulate_rnn_with_task, simulate_rnn_with_input, test_network, get_autonomous_dynamics, get_autonomous_dynamics_from_hinit
from utils import makedirs


##############ANGULAR
def get_manifold_from_closest_projections(trajectories_flat, wo, npoints=128):
    n_rec = wo.shape[0]
    xs = np.arange(-np.pi, np.pi, 2*np.pi/npoints)
    xs = np.append(xs, -np.pi)
    #trajectories_flat = trajectories.reshape((-1,n_rec));
    ys = np.dot(trajectories_flat.reshape((-1,n_rec)), wo)
    circle_points = np.array([np.cos(xs), np.sin(xs)]).T
    dists = cdist(circle_points.reshape((-1,2)), ys)
    csx2 = []
    for i in range(xs.shape[0]):
        csx2.append(trajectories_flat[np.argmin(dists[i,:]),:])
    csx2 = np.array(csx2)
    csx2_proj2 = np.dot(csx2, wo)
    return xs, csx2, csx2_proj2


def get_manifold_from_closest_angular_projections(trajectories_flat, wo, npoints=128):
    n_rec = wo.shape[0]
    xs = np.arange(-np.pi, np.pi, 2*np.pi/npoints)
    xs = np.append(xs, -np.pi)
    #trajectories_flat = trajectories.reshape((-1,n_rec));
    ys = np.dot(trajectories_flat.reshape((-1,n_rec)), wo)
    thetas = np.arctan2(ys[:,0], ys[:,1])
    dists = cdist(xs.reshape((-1,1)), thetas.reshape((-1,1)))
    csx2 = []
    for i in range(xs.shape[0]):
        csx2.append(trajectories_flat[np.argmin(dists[i,:]),:])
    csx2 = np.array(csx2)
    csx2_proj2 = np.dot(csx2, wo)
    return xs, csx2, csx2_proj2


# task = angularintegration_task(T=T, dt=dt, sparsity=1, random_angle_init=False)]
def angular_loss_angint(net, task, h_init, T, batch_size=128, dt=0.1, random_seed=100, noise_std=0.):
    
    if h_init!='random':
        assert h_init.shape[0]==batch_size, "h_init must have same number of points as batch_size"
    
    np.random.seed(random_seed)
    net.noise_std = noise_std
    input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, T, h_init=h_init, batch_size=batch_size)
    output_angle = np.arctan2(output[:,:,1], output[:,:,0]);
    target_angle = np.arctan2(target[:,:,1], target[:,:,0]);
    angle_error = np.abs(output_angle - target_angle)
    angle_error[np.where(angle_error>np.pi)] = 2*np.pi-angle_error[np.where(angle_error>np.pi)]
    mean_error = np.mean(angle_error,axis=0)
    max_error = np.max(angle_error,axis=0)
    
    return mean_error, max_error


def angular_loss_angvel_noinput(net, angle_init, h_init, T, batch_size=128, dt=0.1, random_seed=100, noise_std=0.):
    # if h_init!='random':
    #     assert h_init.shape[0]==batch_size, "h_init must have same number of points as batch_size"
    
    np.random.seed(random_seed)
    net.noise_std = noise_std
    input = np.zeros((angle_init.shape[0], T, net.dims[0]))
    output, trajectories = simulate_rnn_with_input(net, input, h_init=h_init)
    output_angle = np.arctan2(output[:,:,1], output[:,:,0]);
    #target_angle = np.arctan2(angle_init[:,1], angle_init[:,0]);
    angle_error = np.abs(output_angle - angle_init[:, np.newaxis])
    y0 = np.dot(h_init, net.wo.detach().numpy())
    hat_alpha_0 = np.arctan2(y0[:,0], y0[:,1])
    angle_error[:,0] = np.abs(hat_alpha_0 - angle_init)
    angle_error[np.where(angle_error>np.pi)] = 2*np.pi-angle_error[np.where(angle_error>np.pi)]
    #angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

    mean_error = np.mean(angle_error,axis=0)
    max_error = np.max(angle_error,axis=0)
    min_error = np.min(angle_error,axis=0)
    
    eps_mean_int = np.cumsum(mean_error) / np.arange(1,mean_error.shape[0]+1)
    eps_plus_int = np.cumsum(max_error) / np.arange(1,mean_error.shape[0]+1)
    eps_min_int = np.cumsum(min_error) / np.arange(1,mean_error.shape[0]+1)
    
    return min_error, mean_error, max_error, eps_min_int, eps_mean_int, eps_plus_int


def angular_loss_angvel_with_input(net, angle_init, h_init, input, dt=0.1, random_seed=100):
    np.random.seed(random_seed)
    output, trajectories = simulate_rnn_with_input(net, input, h_init=h_init)
    outputs_1d = np.cumsum(input, axis=1)*dt
    output_angle = np.arctan2(output[:,:,1], output[:,:,0]);
    target_angle = angle_init[:, np.newaxis] + outputs_1d.squeeze()
    target_angle = (target_angle + np.pi) % (2 * np.pi) - np.pi
    angle_error = np.abs(output_angle - target_angle)
    angle_error[np.where(angle_error>np.pi)] = 2*np.pi-angle_error[np.where(angle_error>np.pi)]

    mean_error = np.mean(angle_error,axis=0)
    max_error = np.max(angle_error,axis=0)
    min_error = np.min(angle_error,axis=0)
    
    eps_mean_int = np.cumsum(mean_error) / np.arange(mean_error.shape[0])
    eps_plus_int = np.cumsum(max_error) / np.arange(mean_error.shape[0])
    eps_min_int = np.cumsum(min_error) / np.arange(mean_error.shape[0])
    
    return min_error, mean_error, max_error, eps_min_int, eps_mean_int, eps_plus_int


def angle_analysis_on_net(net, T, input_or_task='input',
                          batch_size=128,
                          dt=0.1, 
                          T_inv_man=1e2):
    np.random.seed(100)
    task = angularintegration_task(T=T_inv_man, dt=dt, sparsity=1, random_angle_init=False)
    input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, T_inv_man, h_init='random', batch_size=batch_size)
    xs, csx2, csx2_proj2 = get_manifold_from_closest_projections(trajectories, net.wo.detach().numpy(), npoints=batch_size)

    net.map_output_to_hidden = False
    np.random.seed(100)
    task = angularintegration_task(T=T, dt=dt, sparsity=1, random_angle_init=False)
    input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, T, h_init='random', batch_size=batch_size)
    output_angle = np.arctan2(output[:,:,1], output[:,:,0]);
    target_angle = np.arctan2(target[:,:,1], target[:,:,0]);
    angle_error = np.abs(output_angle - target_angle)
    angle_error[np.where(angle_error>np.pi)] = 2*np.pi-angle_error[np.where(angle_error>np.pi)]
    
    mean_error = np.mean(angle_error,axis=0)
    max_error = np.max(angle_error,axis=0)
    min_error = np.min(angle_error,axis=0)

    eps_mean_int = np.cumsum(mean_error) / np.arange(mean_error.shape[0])
    eps_plus_int = np.cumsum(max_error) / np.arange(mean_error.shape[0])
    eps_min_int = np.cumsum(min_error) / np.arange(mean_error.shape[0])

 
    
    
def angular_loss_msg(net, cue_range, T1_multiple=16, dt=.1, batch_size=128):
    t1_pos = 0 
    time_until_input = 5
    cue_duration = 5
    time_until_cue_range_min, time_until_cue_range_max = cue_range
    min_time = time_until_input + cue_duration + time_until_cue_range_min
    time_until_measured_response = 5
    time_after_response = 2
    add_t = time_until_measured_response+time_after_response
    T1 = time_until_cue_range_max #training_kwargs['time_until_cue_range'][1]
    T = T1*T1_multiple
    tstep=1
    timepoints=int(T)#-15-min_time_until_cue
    print(timepoints)
    
    tstep = 1 #int(T*dt_rnn//10)

    net.noise_std = 0
    net.map_output_to_hidden=False
    np.random.seed(100)
    
    T_tot = T/dt+min_time*10+10
    task = center_out_reaching_task(T=T_tot, dt=dt, time_until_cue_range=[T, T+1], angles_random=False);
    input, target, mask, output, trajectories_full = simulate_rnn_with_task(net, task, T_tot, h_init='random', batch_size=batch_size)
    target = input[:,time_until_input,:]
    target_angle = np.arctan2(target[:,1], target[:,0]);
    target_angle = (target_angle + np.pi) % (2 * np.pi) - np.pi

    angle_errors = np.zeros((batch_size, (timepoints+min_time)//tstep))
    ts = []
    for t in range(min_time,timepoints+min_time,tstep):
        ts.append(t)
        input = np.zeros((batch_size,cue_duration+add_t,3))
        input[:,:cue_duration,2]=1.
        h_init = trajectories_full[:,t,:]
        output, trajectories = simulate_rnn_with_input(net, input, h_init=h_init)
        #ax.plot(output[:,-1,0].T, output[:,-1,1].T, 'b')
        #ax.plot(target[:,0].T, target[:,1].T, 'r')
        output_angle = np.arctan2(output[:,-1,1], output[:,-1,0]);
        
        angle_error = np.abs(output_angle - target_angle)
        angle_error[np.where(angle_error>np.pi)] = 2*np.pi-angle_error[np.where(angle_error>np.pi)]
        mean_error = np.mean(angle_error,axis=0)
        angle_errors[:,(t-min_time)//tstep] = angle_error
        t1_pos=np.min([t1_pos,np.min(mean_error)])

    mean_error = np.mean(angle_errors,axis=0)
    max_error = np.max(angle_errors,axis=0)
    min_error = np.min(angle_errors,axis=0)
    
    eps_mean_int = np.cumsum(mean_error) / np.arange(1, mean_error.shape[0]+1)
    eps_plus_int = np.cumsum(max_error) / np.arange(1, mean_error.shape[0]+1)
    eps_min_int = np.cumsum(min_error) / np.arange(1, mean_error.shape[0]+1)
    
    return ts, eps_min_int, eps_mean_int, eps_plus_int




##########SPEED
def get_speed_and_acceleration(trajectory):
    speeds = []
    accelerations = []
    for t in range(trajectory.shape[0]-1):
        speed = np.linalg.norm(trajectory[t+1, :]-trajectory[t, :])
        speeds.append(speed)
    for t in range(trajectory.shape[0]-2):
        acceleration = np.abs(speeds[t+1]-speeds[t])
        accelerations.append(acceleration)
    speeds = np.array(speeds)
    accelerations = np.array(accelerations)
    return speeds, accelerations

def get_speed_and_acceleration_batch(trajectories):
    all_speeds = []
    all_accs = []
    for trajectory in trajectories:
        speeds, accelerations = get_speed_and_acceleration(trajectory)
        all_speeds.append(speeds)
        all_accs.append(accelerations)
    return np.array(all_speeds), np.array(all_accs)



###slow man
def detect_slow_manifold(trajectories, subsample=100, tol=1e-4):
    all_speeds, all_accs = get_speed_and_acceleration_batch(trajectories)
    idx = np.argmax(all_speeds - all_speeds[:,-1][:,np.newaxis] < tol,axis=1)
    inv_man = np.empty((0,trajectories.shape[-1]))
    for i,index in enumerate(idx):
        inv_man = np.vstack([inv_man, trajectories[i,index::subsample,:]])
        
    return inv_man


##########VFs
def vf_on_ring(trajectories_flat,  wo,  wrec, brec, cs, rnn_ode=tanh_ode,
               fxd_pnt_thetas=None, stabilities=None, method='closest', npoints=128, 
               fig_folder=None, fig_ext=''):
    
    X = []
    Y = []
    U = []
    V = []
    if method=='spline':
        xs = np.arange(-np.pi, np.pi, 2*np.pi/npoints)
        xs = np.append(xs, -np.pi)
        csx=cs(xs);
        # fig, ax = plt.subplots(1, 1, figsize=(3, 3)); 
        diff = np.zeros((csx.shape[0]))

        for i in range(csx.shape[0]):
            x,y=np.cos(xs[i]),np.sin(xs[i])
            x,y=np.dot(wo.T,csx[i])
            u,v=np.dot(wo.T, rnn_ode(0, csx[i], wrec, brec, tau=10))
            diff[i] = np.linalg.norm(np.array([u,v]))

            # plt.quiver(x,y,u,v)
            X.append(x)
            Y.append(y)
            U.append(u)
            V.append(v)

        return diff, np.array(X), np.array(Y), np.array(U), np.array(V)
        # for i,theta in enumerate(fxd_pnt_thetas):
        #     x,y=np.cos(theta),np.sin(theta)
        #     if stabilities[i]==1:
        #         plt.plot(x,y,'.g')
        #     else:
        #         plt.plot(x,y,'.r')
        # plt.axis('off'); fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None);
        # if fig_folder:
        #     plt.savefig(fig_folder+f'/vf_on_ring_{fig_ext}.pdf', bbox_inches="tight")
    
    elif method=='closest':
        #trajectories_flat = trajectories.reshape((-1,trajectories[-1]));
        xs, csx2, csx2_proj2 = get_manifold_from_closest_projections(trajectories_flat, wo, npoints=npoints)
    
        diff = np.zeros((csx2.shape[0]))
        for i in range(csx2.shape[0]):
            x,y=np.cos(xs[i]),np.sin(xs[i])
            x,y=np.dot(csx2[i], wo)
            u,v=np.dot(wo.T, rnn_ode(0, csx2[i], wrec, brec, tau=10))
            diff[i] = np.linalg.norm(np.array([u,v]))
            
            X.append(x)
            Y.append(y)
            U.append(u)
            V.append(v)

        return diff, np.array(X), np.array(Y), np.array(U), np.array(V)
    
    elif method=='points':
        diff = np.zeros((trajectories_flat.shape[0]))

        for i in range(trajectories_flat.shape[0]):
            x,y=np.dot(trajectories_flat[i],wo)
            #x,y=np.cos(xs[i]),np.sin(xs[i])
            #x,y=np.dot(wo.T,csx[i])
            u,v=np.dot(wo.T, rnn_ode(0, trajectories_flat[i], wrec, brec, tau=10))
            diff[i] = np.linalg.norm(np.array([u,v]))

            X.append(x)
            Y.append(y)
            U.append(u)
            V.append(v)
        return diff, np.array(X), np.array(Y), np.array(U), np.array(V)




def plot_vf_on_ring(X,Y,U,V,fxd_pnt_thetas=None,stabilities=None,fig_folder=None,fig_ext='',ax=None,fig=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3)); 
        plt.axis('off'); fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None);

    for x,y,u,v in zip(X,Y,U,V):
        plt.quiver(x,y,u,v)
        
    if np.any(fxd_pnt_thetas):
        for i,theta in enumerate(fxd_pnt_thetas):
            x,y=np.cos(theta),np.sin(theta)
            if stabilities[i]==1:
                plt.plot(x,y,'.g')
            else:
                plt.plot(x,y,'.r')
    if fig_folder:
        #     plt.savefig(fig_folder+f'/vf_on_ring_{fig_ext}.pdf', bbox_inches="tight")
        plt.savefig(fig_folder+f'/vf_on_ring_closest_{fig_ext}.pdf', bbox_inches="tight")
  
def distortion(inv_man_points, wo):
    inv_man_proj2 = np.dot(inv_man_points)
    inv_man_thetas = np.arctan2(inv_man_proj2[:,0], inv_man_proj2[:,1])
    proj_point_on_ring = np.array([np.cos(inv_man_thetas),np.sin(inv_man_thetas)])
    dist = np.np.linalg.norm(inv_man_proj2 - proj_point_on_ring)
    dist_mean = np.mean(dist)
    return dist, dist_mean
    
        
def detect_fixed_points_from_flow_on_ring(trajectories_proj2, cs=None):
    thetas = np.arctan2(trajectories_proj2[:,:,0], trajectories_proj2[:,:,1]);

    thetas_init = thetas[:,0] #np.arange(-np.pi, np.pi, np.pi/batch_size*2);
    idx = np.argsort(thetas_init)
    thetas_init = thetas_init[idx]

    thetas_sorted = thetas[idx]
    theta_unwrapped = np.unwrap(thetas_sorted, period=2*np.pi);
    theta_unwrapped = np.roll(theta_unwrapped, -1, axis=0);
    arr = np.sign(theta_unwrapped[:,-1]-theta_unwrapped[:,0]);
    idx=[i for i, t in enumerate(zip(arr, arr[1:])) if t[0] != t[1]]; 
    stabilities=-arr[idx].astype(int)
    
    fxd_pnt_thetas = thetas_init[idx]
    stab_idx = np.where(stabilities==-1); 
    saddle_idx = np.where(stabilities==1)
    
    if stabilities.shape[0]%2==1:
        fxd_pnt_thetas=np.hstack([fxd_pnt_thetas,0])
        stabilities=np.append(stabilities, -np.sum(stabilities))

    if cs:
        fxd_pnts = cs(fxd_pnt_thetas)
        return fxd_pnt_thetas, stabilities, stab_idx, saddle_idx, fxd_pnts
    else:
        return fxd_pnt_thetas, stabilities, stab_idx, saddle_idx
        
def get_cubic_spline_ring(thetas, invariant_manifold):

    thetas_unique, idx_unique = np.unique(thetas, return_index=True);
    idx_sorted = np.argsort(thetas_unique);
    
    invariant_manifold_unique = invariant_manifold[idx_unique,:];
    invariant_manifold_sorted = invariant_manifold_unique[idx_sorted,:];
    invariant_manifold_sorted[-1,:] = invariant_manifold_sorted[0,:];
    cs = scipy.interpolate.CubicSpline(thetas_unique, invariant_manifold_sorted, bc_type='periodic')
        
    return cs    
        
def vf_inv_man_analysis(inv_man, wo, wrec, brec, cs, rnn_ode):
    
    #get vfs
    vf_diff_spline, X, Y, U_spline, V_spline = vf_on_ring(None, wo,  wrec, brec, cs, method='spline', rnn_ode=rnn_ode)
    vf_diff_closest, X, Y, U_closest, V_closest = vf_on_ring(inv_man, wo,  wrec, brec, None, method='closest', rnn_ode=rnn_ode)
    return vf_diff_spline, vf_diff_closest, U_spline, V_spline, U_closest, V_closest


def tda_trajectories(inv_man, maxdim=1, show=False):
    diagrams = ripser(inv_man, maxdim=maxdim)['dgms']

    if show:
        plot_diagrams(diagrams, show=show)
    return diagrams

    #def get_dimension(inv_man, function=skdim.id.MOM()):
    #    dim=function.fit(inv_man).dimension_
    #    return dim


def analyse_net(net, task, batch_size=128, T=2e4, dt=.1):
    wi, wrec, brec, wo, oth = get_weights_from_net(net)
    # task = center_out_reaching_task(T=T, dt=dt, time_until_cue_range=[T, T+1], angles_random=False);
    
    input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, T, 'random', batch_size=batch_size)
    
    inv_man = detect_slow_manifold(trajectories, subsample=100, tol=1e-4)
    inv_man_proj2 = np.dot(inv_man, wo)
    #dim = get_dimension(inv_man)
    
    inv_man_proj2 = np.dot(inv_man,wo)
    #get angles from invariant manifold
    inv_man_thetas = np.arctan2(inv_man_proj2[:,0], inv_man_proj2[:,1]);

    #fit spline
    cs = get_cubic_spline_ring(inv_man_thetas, inv_man)
    
    xs, csx2, csx2_proj2 = get_manifold_from_closest_projections(inv_man, wo, npoints=batch_size);
    rnn_ode = get_rnn_ode(net.nonlinearity)
    vf_diff_spline, vf_diff_closest = vf_inv_man_analysis(inv_man, wo, wrec, brec, cs, rnn_ode=rnn_ode)
    
    
def s_type_perturbations(net, h_init, timesteps, perturbations_per_h0=1, noise_std=1e-3):
    
    net.map_output_to_hidden = False
    s_perts = np.random.normal(0, noise_std, (h_init.shape[0]*perturbations_per_h0, net.dims[1]))
    h_init_ = np.tile(h_init, (perturbations_per_h0, 1, 1)).reshape((-1, net.dims[1]))
    h_init_ += s_perts
    
    input = np.zeros((h_init_.shape[0], timesteps, net.dims[0]))
    output, trajectories = simulate_rnn_with_input(net, input, h_init_)
    
    trajectories_flat = trajectories.reshape((-1,net.dims[1]));

    d = cdist(trajectories_flat,h_init)
    dist_to_inv_man = np.min(d,axis=1).reshape((timesteps*perturbations_per_h0,-1));
    
    return output, trajectories, dist_to_inv_man


def do_all_angular_analysis(folder, T1, batch_size=128, T1_multiple=16, speed_range=[0.05,0.05]):
    
    #main_exp_name='/angular_integration/T25_Mml_rnn_Stanh_N512_Ihighgain_R0/'
    #folder = parent_dir+"/experiments/" + main_exp_name
    
    # params_path = glob.glob(folder + '/param*.yml')[0]
    # with open(params_path, 'rb') as f:
    #     training_kwargs = pickle.load(f)
    # T1 = T#training_kwargs['T']#/training_kwargs['dt_task']
    T=int(T1*T1_multiple); dt=.1;#dt=training_kwargs['dt_rnn'];
    exp_list = glob.glob(folder + "/res*")
    nexps = len(exp_list)
    all_eps = np.empty((nexps,3,T))
    for exp_i in range(len(exp_list)):
        path = exp_list[exp_i]
        net, result = load_net_path(path)
        net.map_output_to_hidden = False
        wi, wrec, brec, wo, oth = get_weights_from_net(net);
        #training_kwargs = result['training_kwargs'] #

        task = angularintegration_task_constant(T=T, dt=dt, speed_range=speed_range, sparsity=1, random_angle_init='equally_spaced');
        input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, T, 'random', batch_size=batch_size)
        xs, csx2, csx2_proj2 = get_manifold_from_closest_projections(trajectories.reshape((-1,net.dims[1])), net.wo.detach().numpy(), npoints=batch_size);
        xs, csx2, csx2_proj2 = get_manifold_from_closest_angular_projections(trajectories.reshape((-1,net.dims[1])), net.wo.detach().numpy(), npoints=batch_size);

        #eps_min_int, eps_mean_int, eps_plus_int 
        all_eps[exp_i] = angular_loss_angvel_noinput(net, xs, csx2, T, batch_size=csx2.shape[0])
        
        
    np.save(folder+'all_eps.npy', all_eps)




##################

def plot_losses(folder):
    paths = glob.glob(folder + "/result*")
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 3));
    all_losses = np.empty((100,5000))
    all_losses[:] = np.nan
    for exp_i, path in tqdm(enumerate(paths)):
        net, result = load_net_path(path)
        losses = result['losses']
        ax.plot(losses[:np.argmin(losses)], 'b', alpha=.05)
        all_losses[exp_i, :np.argmin(losses)-1] = losses[:np.argmin(losses)-1]
    ax.plot(np.nanmean(all_losses,axis=0), 'b', zorder=1000, label='tanh')
    
    
def plot_losses_in_folder(folder):
    #main_exp_name='center_out/variable_N100_T250_Tr100/tanh/'
    #folder = parent_dir+"/experiments/" + main_exp_name
    exp_list = glob.glob(folder + "/res*")
    nexps = len(exp_list)
    all_losses = np.empty((nexps,10000))
    all_losses[:] = np.nan
    fig, axs = plt.subplots(1, 2, figsize=(3, 3))
    for exp_i in range(nexps):
        # print(exp_i)
        path = exp_list[exp_i]
        net, result = load_net_path(path)
        # wi, wrec, brec, wo, oth = get_weights_from_net(net);
        losses = result['losses']
        all_losses[exp_i,:np.argmin(losses)-1] = losses[:np.argmin(losses)-1].copy()
        axs[0].plot(losses[:np.argmin(losses)].copy(), 'b',  alpha=.01); 
    axs[0].plot(np.nanmean(all_losses,axis=0), 'b', zorder=1000)
    last_non_nan_idx = np.argmax(np.isnan(all_losses), axis=1)
    last_non_nan_idx = np.maximum(0, last_non_nan_idx - 1)
    last_non_nan = all_losses[np.arange(all_losses.shape[0]), last_non_nan_idx]

    bins = np.logspace(np.log10(np.min(last_non_nan)), np.log10(np.max(last_non_nan)), 20)
    axs[1].hist(last_non_nan, bins=bins, orientation='horizontal')

    axs[0].set_yscale('log'); axs[1].set_yscale('log')
    axs[0].set_ylim([np.nanmin(all_losses), np.nanmax(all_losses)])
    axs[1].set_ylim([np.nanmin(all_losses), np.nanmax(all_losses)])
    axs[1].axis('off')
    
    
def nonlinearity_to_marker_mapping(nonlinearity):
    if nonlinearity=='rect_tanh':
        return 'o'  # Circle marker
    elif nonlinearity=='tanh':
        return 's'  # Square marker
    elif nonlinearity=='relu':
        return '^'  # Triangle marker
    
def size_to_color_mapping(nrec):
    if nrec==64:
        return 'b' 
    elif nrec==128:
        return 'g' 
    elif nrec==256:
        return 'orange'
    
nrecs=[64,128,256]
colors=[size_to_color_mapping(nrec) for nrec in nrecs]

def plot_example_trajectories_msg(trajectories_proj2, xs, folder):
    
    cmap = plt.get_cmap('hsv')
    norm = mpl.colors.Normalize(-np.pi, np.pi)
    
    fig, ax = plt.subplots(1, 1, figsize=(3, 3));
    for i in range(trajectories_proj2.shape[0]):
        plt.plot(trajectories_proj2[i,:,0].T, trajectories_proj2[i,:,1].T, color=cmap(norm(xs[i])));
    plt.scatter(np.cos(xs), np.sin(xs), s=100, facecolors='none', edgecolors=cmap(norm(xs)))
    ax.axis('off')
    plt.savefig(folder+'example_trials_2d.pdf', bbox_inches="tight");
    
    
def plot_spline_vs_nmse(df, figfolder):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3));
    #ax.plot(df2['vf_infty_spline'], db(df2['mse_normalized']),'x'); plt
    ax.set_xscale('log'); ax.set_xlabel('$|f|_\infty$');ax.set_ylabel('NMSE (dB)'); 
    plt.axhline(-20, color='red', linestyle='--')
    ax = plt.gca()
    for nonlinearity in df['S'].unique():
        subset = df[df['S'] == nonlinearity]
        for nrec in df['N'].unique(): 
            subsubset = subset[subset['N']==nrec]
            plt.scatter(subsubset['vf_infty_closest'], decibel(subsubset['mse_normalized']), color=size_to_color_mapping(nrec), marker=nonlinearity_to_marker_mapping(nonlinearity))
        #plt.scatter(subset['vf_infty_spline'], db(subset['mse_normalized']), marker=nonlinearity_to_marker_mapping(nonlinearity), label=nonlinearity)
    first_legend = plt.legend(title='Nonlinearity', loc='upper left')
    lines = [Line2D([0], [0], color=c, linewidth=2, linestyle='-') for c in colors]
    first_legend = plt.legend(lines, nrecs, title='Network Size', loc='lower right')
    ax.add_artist(first_legend)
    marker_legend = [Line2D([0], [0], color='black', marker=m, linestyle='') for m in markers]
    plt.legend(marker_legend, df['S'].unique(), title='nonlinearity', loc='upper right')
    plt.savefig(figfolder+"/vfinftyclosest_nmse.pdf");


def plot_vf_vs_meanangularerror(df, figfolder):
    for nonlinearity in df['S'].unique():
        subset = df[df['S']==nonlinearity]

        for nrec in df['N'].unique():
            subsubset = subset[df['N']==nrec]
            vf_infty = np.stack(subsubset['vf_infty_spline'].to_numpy())
            mean_error_0 = np.stack(subsubset['mean_error_0'].to_numpy())
            mean_error_0_infty = mean_error_0[:,128]
            mean_error_0_infty = mean_error_0[:,-1]
            marker=nonlinearity_to_marker_mapping(nonlinearity)
            color=size_to_color_mapping(nrec)
            plt.scatter(vf_infty,mean_error_0_infty,color=color,marker=marker)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$|f|_\infty$')
    plt.ylabel('mean angular error')
    plt.savefig(figfolder+"/vfinftyspline_vs_meanangularerror.pdf");
    
    
def plot(df):
    for nonlinearity in df['S'].unique():
        subset = df[df['S']==nonlinearity]
        for nrec in df['N'].unique():
            subsubset = subset[df['N']==nrec]
            vf_infty = np.stack(subsubset['vf_infty_closest'].to_numpy())
            mean_error_0 = np.stack(subsubset['mean_error_0'].to_numpy())
            eps_mean_int = np.cumsum(mean_error_0,axis=1) / np.arange(1, mean_error_0.shape[1]+1)[None, :]
            mean_error_0_t1 = eps_mean_int[:,128]
            mean_error_0_infty = eps_mean_int[:,-1]
            marker=nonlinearity_to_marker_mapping(nonlinearity)
            color=size_to_color_mapping(nrec)
            plt.scatter(vf_infty,mean_error_0_t1,color=color,marker=marker)
            plt.scatter(vf_infty,mean_error_0_infty,color=color,marker=marker,facecolors='none', edgecolors=color)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$|f|_\infty$')
    plt.ylabel('mean angular error')
    
def plot_the_high_and_the_low(df, T1, figfolder):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3));
    
    row_lownfps = df.iloc[np.where(df['nfps_csx2']==6.)[0][0]]
    mean_error_0 = row_lownfps['mean_error_0'][T1:int(T1*10)]
    #max_error_0 = row_lownfps['max_error_0'][T1:int(T1*10)]
    plt.plot(mean_error_0,color='b')
    #plt.plot(max_error_0,'--', color='b')
    plt.plot(9.5*T1, mean_error_0[-1], '.', color='b');
    #plt.plot(9.5*T1, max_error_0[-1], '.', color='b');
    row_highnfps = df.iloc[np.where(df['nfps_csx2']==40.)[0][0]]
    mean_error_0 = row_highnfps['mean_error_0'][T1:int(T1*10)]
    #max_error_0 = row_highnfps['max_error_0'][T1:int(T1*10)]
    plt.plot(mean_error_0,color='orange')
    #plt.plot(max_error_0,'--', color='orange')
    plt.plot(9.5*T1, mean_error_0[-1], '.', color='orange');
    
    x = np.arange(T1,10.5*T1,T1/2.)
    y = row_lownfps['vf_infty_closest']*x
    plt.plot(x-T1,y,'--', color='orange')
    y = row_highnfps['vf_infty_closest']*x
    plt.plot(x-T1,y,'--', color='b')
    
    plt.xlabel("t")
    plt.ylabel("mean angular error")
    ax.set_xticks(np.append(np.arange(0,10*T1,2*T1),9.5*T1),['$T_1$']+[f'${i}T_1$' for i in range(2,10,2)]+['$\infty$'])
    
    lines = [Line2D([0], [0], color=c, linewidth=2, linestyle='-') for c in ['b', 'orange']]
    first_legend = plt.legend(lines, np.array(nrecs)[[0,2]], title='Network Size', loc='lower right')
    ax.add_artist(first_legend)
    
    lines = [Line2D([0], [0], color='k', linewidth=2, linestyle=ls) for ls in ['-', '--']]
    second_legend = plt.legend(lines, np.array(nrecs)[[0,2]], title='Network Size', loc='lower right')
    plt.legend(lines, ['mean angular error',r'$\|\varphi\|_\infty$'], title='', loc='upper right')
    
    plt.savefig(figfolder+"/meanangularerror_lowhigh.pdf", bbox_inches="tight");

def max_angle(fxd_pnt_thetas):
    dist_next = fxd_pnt_thetas-np.roll(fxd_pnt_thetas,1)
    dist_next[np.where(dist_next>np.pi)] = 2*np.pi-dist_next[np.where(dist_next>np.pi)]
    dist_next = (dist_next + np.pi) % (2 * np.pi) - np.pi

    return np.max(dist_next)

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


def analysis(folder, df, batch_size=128, T=256, T1_multiple=16, auton_mult=4, input_strength=0.05, subsample=10, tol=1e-2):
    #main_exp_name='/experiments/angular_integration_old/N64_T128_noisy/relu/';
    #folder=parent_dir+main_exp_name
    exp_list = glob.glob(folder + "/res*")
    nexps = len(exp_list)
    
    for exp_i in range(nexps):
        path = exp_list[exp_i]
        print(path)        
        net, result = load_net_path(path)
        training_kwargs = result['training_kwargs']
        losses = result['losses']
        net.noise_std = 0
        wi, wrec, brec, wo, oth = get_weights_from_net(net)
        mse, mse_normalized = test_network(net)
        output, trajectories_0 = get_autonomous_dynamics(net, T=T, dt=.1, batch_size=batch_size)
        
        net.map_output_to_hidden = False
        inv_man = detect_slow_manifold(trajectories_0, subsample=subsample, tol=tol)
        #inv_man_proj2 = np.dot(inv_man, wo)
        
        #dim = get_dimension(inv_man)
        #print("MSE: ", mse, "dB: ", db(normalized_mse)) #, "D:", dim)
        xs, csx2, csx2_proj2 = get_manifold_from_closest_projections(inv_man, wo, npoints=batch_size);
        plt.plot(csx2_proj2[:,0], csx2_proj2[:,1], '.')
        
        #fit spline
        inv_man_thetas = np.arctan2(csx2_proj2[:,0], csx2_proj2[:,1]);
        cs = get_cubic_spline_ring(inv_man_thetas, csx2)
        
        output, trajectories = get_autonomous_dynamics_from_hinit(net, trajectories_0[:,100,:], T=int(T*auton_mult))
        fxd_pnt_thetas_traj, stabilities_traj, stab_idx, saddle_idx, fxd_pnts = detect_fixed_points_from_flow_on_ring(output, cs=cs)
        plt.plot(np.cos(fxd_pnt_thetas_traj), np.sin(fxd_pnt_thetas_traj), '.b')
        nfps_traj = fxd_pnt_thetas_traj.shape[0]
        maxangle_traj = max_angle(fxd_pnt_thetas_traj)
        boa_traj = boa(fxd_pnt_thetas_traj)
        
        output, trajectories = get_autonomous_dynamics_from_hinit(net, csx2, T=int(T/2))
        fxd_pnt_thetas_csx2, stabilities_csx2, stab_idx, saddle_idx, fxd_pnts = detect_fixed_points_from_flow_on_ring(output, cs=cs)
        plt.plot(np.cos(fxd_pnt_thetas_csx2), np.sin(fxd_pnt_thetas_csx2), '.r')
        nfps_csx2 = fxd_pnt_thetas_csx2.shape[0]
        maxangle_csx2 = max_angle(fxd_pnt_thetas_csx2)
        boa_csx2 = boa(fxd_pnt_thetas_csx2)
        
        rnn_ode=get_rnn_ode(training_kwargs['nonlinearity'])
        vf_diff_spline, vf_diff_closest, U_spline, V_spline, U_closest, V_closest = vf_inv_man_analysis(inv_man, wo, wrec, brec, cs, rnn_ode=rnn_ode)
        #vf_diff_points, X, Y, U, V = vf_on_ring(csx2, wo,  wrec, brec, None, method='points', rnn_ode=rnn_ode)
        
        min_error_0, mean_error_0, max_error_0, eps_min_int_0, eps_mean_int_0, eps_plus_int_0 = angular_loss_angvel_noinput(net, angle_init=xs, h_init=csx2, T=T*T1_multiple, batch_size=batch_size, dt=0.1, random_seed=100, noise_std=0.)

        input = input_strength*np.ones((csx2.shape[0], T*T1_multiple, net.dims[0]));
        min_error_input, mean_error_input, max_error_input, eps_min_int_input, eps_mean_int_input, eps_plus_int_input = angular_loss_angvel_with_input(net, angle_init=xs, h_init=csx2, input=input)
        
        T1, N, I, S, R, M, clip_gradient = get_tr_par(training_kwargs)
        df = df.append({'path':path,
        'T': T1, 'N': N, 'I': I, 'S': S, 'R': R, 'M': M, 'clip_gradient':clip_gradient,
        'losses': losses,
                    'trial': exp_i,
                    'mse': mse,
                    'mse_normalized':mse_normalized,
                    'nfps_traj': nfps_traj,
                    'nfps_csx2': nfps_csx2,
                    'stabilities_traj':stabilities_traj,
                    'stabilities_csx2':stabilities_csx2,
                    'fxd_pnt_thetas_traj':fxd_pnt_thetas_traj,
                    'fxd_pnt_thetas_csx2':fxd_pnt_thetas_csx2,
                    'maxangle_traj':maxangle_traj,
                    'maxangle_csx2':maxangle_csx2,
                    'boa_traj':boa_traj,
                    'boa_csx2':boa_csx2,
                    'vf_diff_closest':vf_diff_closest,
                     'vf_diff_spline':vf_diff_spline,
                     'vf_infty_closest':np.max(vf_diff_closest),
                     'vf_infty_spline':np.max(vf_diff_spline),
                     'vf_closest':np.array([U_closest, V_closest]),
                     'vf_spline':np.array([U_spline, V_spline]),
                     'inv_man':inv_man,
                      'min_error_0':min_error_0,
                       'mean_error_0':mean_error_0,
                        'max_error_0':max_error_0,
                        'eps_min_int_0':eps_min_int_0,
                        'eps_mean_int_0':eps_mean_int_0,
                        'eps_plus_int_0':eps_plus_int_0,
                        'min_error_input': min_error_input,
                        'mean_error_input':mean_error_input,
                        'max_error_input':max_error_input,
                        'eps_min_int_input':eps_min_int_input,
                        'eps_mean_int_input':eps_mean_int_input,
                        'eps_plus_int_input':eps_plus_int_input
                        }, ignore_index=True)
        
    return df
        
        
def all_analysis(folders):
    df = pd.DataFrame(columns=['path', 'T', 'N', 'I', 'S', 'R', 'M', 'trial', 'mse',
                               'normalized_mse', 'vf_diff_closest', 'vf_diff_spline',
                               'eps_min_0', 'eps_mean_0', 'eps_plus_0',
                               'eps_min_input', 'eps_mean_input', 'eps_plus_input', 'losses'])

    for folder in folders:
        df = analysis(folder, df)
        
    return df



markers=['o', 's', '^']
nrecs=[64,128,256]
nonlins=['relu','tanh','recttanh'];
folders = [parent_dir + f'/experiments/angular_integration_old/N{nrec}_T128_noisy/{nonlin}/' 
           for nrec in nrecs for nonlin in nonlins]