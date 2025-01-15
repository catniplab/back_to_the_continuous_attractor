# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:20:28 2024

@author: abel_
"""
    
import os, sys
import glob
import pickle
import yaml
from pathlib import Path, PurePath
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir) 

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from angular_loss_analysis import load_net_path, simulate_rnn_with_task, get_rnn_ode
from lstm import nmse
from analysis_functions import find_periodic_orbits
from tasks import double_angularintegration_task
from odes import recttanh_jacobian_point, tanh_jacobian, relu_jacobian
from utils import makedirs

def run_all_analysis():
    df = pd.DataFrame(columns=['path', 'T', 'N'])

    nrecs=[64,128,256]; nonlins=['relu','tanh','recttanh']; 
    for N in nrecs:
        for nonlin in nonlins:
            df = run_analysis_persubexp(df, N=N, nonlinearity=nonlin)

    return df

def run_analysis_persubexp(df, N=64, nonlinearity='recttanh',
                     epsilon=.1):
    exp_path = parent_dir + f'/experiments/double_angular/N{N}_T128/{nonlinearity}/'
    print(exp_path)

    h_init='random'
    T=12.8*5; dt=.1;
    task = double_angularintegration_task(T=T, dt=dt, sparsity=1, random_angle_init='equally_spaced', speed_range=[0.,0.], constant_speed=True);
    batch_size = 256*4
    pca = PCA(n_components=10)
    params_path = glob.glob(exp_path + '/param*.yml')[0];
    training_kwargs = yaml.safe_load(Path(params_path).read_text());
    rnn_ode = get_rnn_ode(training_kwargs['nonlinearity'])
    rnn_ode_jacobian = get_rnn_jacobian(training_kwargs['nonlinearity'])

    exp_list = glob.glob(exp_path + "/res*")
    for exp_i in range(10):
        exp = exp_list[exp_i]
        net, result = load_net_path(exp, which='post')
        n_rec = net.dims[1]
        wi, wrec, wo, brec, h0, oth = result['weights_last']
        net.noise_std=0;
        input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, 12.8*2, h_init, batch_size);
        mse, mse_normalized, mse_db = nmse(target, output, from_t=0, to_t=None)
        if mse_db>-3:
            continue
        
        input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, T, h_init, batch_size);
        output_angle1 = np.arctan2(output[...,1], output[...,0]);
        output_angle2 = np.arctan2(output[...,3], output[...,2]);
        fxd_pnts = output[:,-1,:]
        # db = DBSCAN(eps=epsilon, min_samples=1).fit(fxd_pnts);
        # unique_indices = np.unique(db.labels_, return_index=True)[1]
        # unique_points = fxd_pnts[unique_indices]; 
        # _, _, _, _, trajectories_start = simulate_rnn_with_task(net, task, 1, h_init, batch_size*1); = np.arctan2(unique_points[...,1], unique_points[...,0]);
        # unique_angle2 = np.arctan2(unique_points[...,3], unique_points[...,2]);
        # print(unique_angle1.shape[0])
        trajectories_flat = trajectories[:,30:500:100,:].reshape((-1,trajectories.shape[-1]));
        trajectories_pca = pca.fit_transform(trajectories_flat);
        trajectories_pca_time = trajectories_pca.reshape((batch_size,-1,10))
        recurrences, recurrences_pca = find_periodic_orbits(trajectories, trajectories_pca_time, limcyctol=1e-2, mindtol=1e-10)
        fxd_pnts = np.array([recurrence[0] for recurrence in recurrences if len(recurrence)==1 or len(recurrence)==11]).reshape((-1,n_rec));
        #lcs =[recurrence for recurrence in recurrences if len(recurrence)!=1 and len(recurrence)>11];
        if fxd_pnts.shape[0]==0:
            continue
        db = DBSCAN(eps=epsilon, min_samples=1).fit(fxd_pnts);
        unique_indices = np.unique(db.labels_, return_index=True)[1]
        fps = fxd_pnts[unique_indices]
        fps_out = np.dot(fps, wo)
        nfps = unique_indices.shape[0]
        stabilities = stabilities_fps(wrec, brec, fps, rnn_ode_jacobian)
        fxd_pnt_thetas1 = np.arctan2(fps_out[...,1], fps_out[...,0]);
        fxd_pnt_thetas2 = np.arctan2(fps_out[...,3], fps_out[...,2]);
        print(exp_i, nfps, mse_db)
        fp_boa = double_boa(fxd_pnt_thetas1, fxd_pnt_thetas2)
        mean_fp_dist = double_mean_fp_distance(fxd_pnt_thetas1, fxd_pnt_thetas2)
        
        trajectories_start = trajectories[:,0,:]
        vf_infty, uni_norm1, uni_norm2, diff, diff1, diff2 = get_vf_norm(trajectories_start, wrec, brec, wo, rnn_ode)
        
        eigenspectrum = eigenspectrum_invman(wrec, brec, trajectories_start, rnn_ode_jacobian)
        
        min_error, mean_error, max_error, eps_min_int, eps_mean_int, eps_plus_int = angular_double_error(target, output)

        df = df.append({'path':exp, 'S':nonlinearity,
        'T': T, 'N': N, 'scale_factor': .5,  'dropout': 0., 'M': True, 'clip_gradient':1,
                    'trial': exp_i,
                    'mse': mse,
                    'mse_normalized':mse_normalized,
                    'nfps': nfps,
                    'fps':fps,
                    'fps_out':fps_out,
                    'stabilities':stabilities,
                    'boa':fp_boa,
                    'mean_fp_dist':mean_fp_dist,
                    'inv_man':trajectories[:,0,:],
                     'inv_man_output':output[:,0,:],
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


# plt.hist(max_eigv,color='b',alpha=.5, density=True);plt.hist(max_eigv_2nd,color='purple',alpha=.5, density=True);plt.hist(eigv_rest,color='k',alpha=.5, density=True);

# max_eigv = []
# max_eigv_2nd = []
# for i,x in enumerate(init_states[::16,:]):
#     J = recttanh_jacobian_point(wrec,brec,1,x)
#     eigenvalues, eigenvectors = np.linalg.eig(J)
#     eigenvalues = sorted(np.real(eigenvalues))
#     max_eigv.append(eigenvalues[-1])
#     max_eigv_2nd.append(eigenvalues[-2])
#     plt.scatter([i]*n_rec, eigenvalues, s=1, c='k', marker='o', alpha=0.5); 
# plt.plot(max_eigv, 'b', label='1st')
# plt.plot(max_eigv_2nd, 'purple', label='2nd')
# plt.xlabel(r'$\theta$')
# plt.ylabel('eigenvalue spectrum')
# plt.xticks([0,len(csx)], [0,r'$2\pi$'])
# plt.ylim([-1.5,0.2])
# plt.legend(loc='lower right')
# plt.hlines(0, 0,len(csx), 'r', linestyles='dotted');
# plt.savefig(exp_path+f'/eigenvalue_spectrum_{exp_i}.pdf')




def double_boa(fxd_pnt_thetas1, fxd_pnt_thetas2):
    fxd_pnt_thetas1 = np.sort(fxd_pnt_thetas1)
    fxd_pnt_thetas2 = np.sort(fxd_pnt_thetas2)
    nfps = fxd_pnt_thetas1.shape[0]
    boas1 = []
    for i in range(0,nfps-1,2):
        boa = (fxd_pnt_thetas1[i+1]-fxd_pnt_thetas1[i-1])% (2*np.pi)
        boas1.append(boa/np.pi/2)
    if nfps>2:
        perf1 = -np.sum([boa*np.log(boa) for boa in boas1])
    else:
        perf1 = 0
        
    boas2 = []
    for i in range(0,nfps-1,2):
        boa = (fxd_pnt_thetas2[i+1]-fxd_pnt_thetas2[i-1])% (2*np.pi)
        boas2.append(boa/np.pi/2)
    if nfps>2:
        perf2 = -np.sum([boa*np.log(boa) for boa in boas2])
    else:
        perf2 = 0
    return perf1+perf2


def angular_double_error(target, output):

    output_angle1 = np.arctan2(output[:,:,1], output[:,:,0]);
    target_angle1 = np.arctan2(target[:,:,1], target[:,:,0]);
    angle_error1 = np.abs(output_angle1 - target_angle1)
    angle_error1[np.where(angle_error1>np.pi)] = 2*np.pi-angle_error1[np.where(angle_error1>np.pi)]
    
    output_angle2 = np.arctan2(output[:,:,3], output[:,:,2]);
    target_angle2 = np.arctan2(target[:,:,3], target[:,:,2]);
    angle_error2 = np.abs(output_angle2 - target_angle2)
    angle_error2[np.where(angle_error2>np.pi)] = 2*np.pi-angle_error2[np.where(angle_error2>np.pi)]

    angle_error = angle_error1+angle_error2
    mean_error = np.mean(angle_error,axis=0)
    max_error = np.max(angle_error,axis=0)
    min_error = np.min(angle_error,axis=0)

    eps_mean_int = np.cumsum(mean_error) / np.arange(1,1+mean_error.shape[0])
    eps_plus_int = np.cumsum(max_error) / np.arange(1,1+mean_error.shape[0])
    eps_min_int = np.cumsum(min_error) / np.arange(1,1+mean_error.shape[0])
    
    return min_error, mean_error, max_error, eps_min_int, eps_mean_int, eps_plus_int



#eigenspectrum
def get_rnn_jacobian(nonlinearity):
    if nonlinearity == 'tanh':
        return relu_jacobian
    elif nonlinearity == 'relu':
        return tanh_jacobian
    elif nonlinearity == 'rect_tanh':
        return recttanh_jacobian_point

def eigenspectrum_invman(wrec, brec, csx, rnn_ode_jacobian):
    eigenspectrum = []
    for i,x in enumerate(csx):
        J = rnn_ode_jacobian(wrec,brec,1,x)
        eigenvalues, eigenvectors = np.linalg.eig(J)
        eigenvalues = sorted(np.real(eigenvalues))
        eigenspectrum.append(eigenvalues)
    return eigenspectrum

def stabilities_fps(wrec, brec, fps, rnn_ode_jacobian):
    stabilities = []
    for i,x in enumerate(fps):
        J = rnn_ode_jacobian(wrec,brec,1,x)
        eigenvalues, eigenvectors = np.linalg.eig(J)
        eigenvalues = sorted(np.real(eigenvalues))
        if eigenvalues[-1]>0:
            stabilities.append(1)
        else:
            stabilities.append(-1)
    return stabilities

def double_mean_fp_distance(fxd_pnt_thetas1, fxd_pnt_thetas2):
    if fxd_pnt_thetas1.shape[0]==0:
        return np.inf
    pairwise_distances1 = np.diff(fxd_pnt_thetas1)
    pairwise_distances1 = np.mod(pairwise_distances1, 2*np.pi)
    
    if fxd_pnt_thetas2.shape[0]==0:
        return np.inf
    pairwise_distances2 = np.diff(fxd_pnt_thetas2)
    pairwise_distances2 = np.mod(pairwise_distances2, 2*np.pi)
    
    return np.mean(pairwise_distances1)+np.mean(pairwise_distances1)

################VF uniform norm
# rnn_ode = get_rnn_ode(training_kwargs['nonlinearity'])
def get_vf_norm(trajectories_start_flat, wrec, brec, wo, rnn_ode):
    diff = np.zeros((trajectories_start_flat.shape[0]))
    diff1 = np.zeros((trajectories_start_flat.shape[0]))
    diff2 = np.zeros((trajectories_start_flat.shape[0]))
    for i in range(trajectories_start_flat.shape[0]):
        x,y,z,a=np.dot(wo.T,trajectories_start_flat[i,:])
        u,v,n,m=np.dot(wo.T, rnn_ode(0, trajectories_start_flat[i], wrec, brec, tau=10))
        diff[i] = np.linalg.norm(np.array([u,v,n,m]))
        
        diff1[i] = np.linalg.norm(np.array([u,v]))
        diff2[i] = np.linalg.norm(np.array([n,m]))
    uni_norm_tot = np.max(diff)
    uni_norm1 = np.max(diff1)
    uni_norm2 = np.max(diff2)
    return uni_norm_tot, uni_norm1, uni_norm2, diff, diff1, diff2


#####################FPS

# fig = plt.figure(figsize=(3,3))
# ax = fig.add_subplot(111); 
# ax.grid(True)
# #ax.scatter(theta[:,:],phi[:,:],s=1); 
# #ax.scatter(angle1[:,-10:],angle2[:,-10:],s=5,c='orange',alpha=0.5)
# max_eigs = []
# for fp in unique_points:
#     rec_out = np.dot(np.array(fp),wo)
#     rec_angle1 = np.mod(np.arctan2(rec_out[...,1], rec_out[...,0]),2*np.pi)
#     rec_angle2 = np.mod(np.arctan2(rec_out[...,3], rec_out[...,2]),2*np.pi)

#     J = recttanh_jacobian_point(wrec,brec,1,fp)    
#     eigenvalues, eigenvectors = np.linalg.eig(J)
#     eigenvalues = sorted(np.real(eigenvalues))
#     max_eigs.append(eigenvalues[-1])
#     if eigenvalues[-1]>0.:
#         ax.scatter(rec_angle1,rec_angle2, c='r', s=5, alpha=1,zorder=10)
#     else:
#         ax.scatter(rec_angle1,rec_angle2, c='g', s=5, alpha=1,zorder=10)
    
# ticks = np.linspace(0, 2 * np.pi, 9)  # 9 ticks from 0 to 2π
# ax.set_xticks(ticks)
# ax.set_xticklabels(['0' if tick == 0 else r'$2\pi$' if tick == 2 * np.pi else '' for tick in ticks])
# ax.set_yticks(ticks)
# ax.set_yticklabels(['0' if tick == 0 else r'$2\pi$' if tick == 2 * np.pi else '' for tick in ticks])
# plt.savefig(exp_path+'/output_2D_fps.pdf');



# fig = plt.figure(figsize=(6,3))
# ax = fig.add_subplot(121); 
# for i in range(batch_size):
#     ax.plot(trajectories_pca_time[i,:,0], trajectories_pca_time[i,:,2],'k',zorder=-10,alpha=.3);
# ax.scatter(trajectories_pca_time[:,-1,0], trajectories_pca_time[:,-1,2],s=1);
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax = fig.add_subplot(122);
# for i in range(batch_size):
#     ax.plot(trajectories_pca_time[i,:,1], trajectories_pca_time[i,:,3],'k',zorder=-10,alpha=.3);
# ax.scatter(trajectories_pca_time[:,-1,1], trajectories_pca_time[:,-1,3],s=1);
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.yaxis.set_major_locator(MaxNLocator(integer=True)); plt.savefig(exp_path+'/torus_exp0_02_13.pdf');