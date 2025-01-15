# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 18:58:02 2023

@author: 
"""

import glob
import os, sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir) 

import torch
import pickle
import yaml
from pathlib import Path, PurePath
import numpy as np
import pandas as pd
from numpy.linalg import svd 
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from itertools import chain, combinations, permutations
from tqdm import tqdm
import sklearn
from scipy.spatial.distance import pdist, squareform
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib.colors as mplcolors
import matplotlib.cm as cmx
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from matplotlib import transforms
from pylab import rcParams
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from models import RNN, run_net, LSTM_noforget, run_lstm
from tasks import angularintegration_task, angularintegration_delta_task, simplestep_integration_task, poisson_clicks_task, center_out_reaching_task, contbernouilli_noisy_integration_task
from analysis_functions import calculate_lyapunov_spectrum, participation_ratio, identify_limit_cycle, find_periodic_orbits, find_analytic_fixed_points, powerset, find_slow_points
from odes import tanh_jacobian, relu_step_input, ReLU, relu_ode
from load_network import get_params_exp, load_net_from_weights, load_all
from simulate_network import simulate_rnn
from utils import makedirs
        

model_names=['lstm', 'low','high', 'ortho', 'qpta']
mean_colors = ['k', 'r',  'darkorange', 'g', 'b']
trial_colors = ['gray', 'lightcoral', 'moccasin', 'lime', 'deepskyblue']
labels=['LSTM', r'$g=.5$', r'$g=1.5$', 'Orthogonal', 'QPTA']
        
def set_3daxes(ax):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3", rotation=90)
    ax.zaxis.labelpad=-15 # <- change the value here
    ax.grid(False)
    return ax


def plot_io_net(net, task, ax=None, ax2=None):
    np.random.seed(12345)
    input, target, mask, output, loss = run_net(net, task, batch_size=1, return_dynamics=False, h_init=None)
    
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax2 = ax.twinx()
    ax2.plot(output[0,:,0], '--', color='deepskyblue', alpha=0.5)
    ax2.plot(output[0,:,1], '--', color='lightcoral', alpha=0.5)
    ax.plot(input[0,...], 'k', label='Angular valocity')
    ax2.plot(target[0,:,0], 'b', label='Target cos')
    ax2.plot(target[0,:,1], 'r', label='Target sin')
    ax2.plot(output[0,:,0], '--', color='deepskyblue', alpha=0.5, label='Output cos')
    ax2.plot(output[0,:,1], '--', color='lightcoral', alpha=0.5, label='Output sin')
    
    ax.set_ylabel("Angular velocity")
    ax.set_xlabel("Time")
    ax2.set_ylabel("Head direction")
    # ax2.legend()
    
    return ax, ax2

def plot_io(ax, ax2, input, output, target):
    ax.plot(input[0,...], 'k', label='Angular valocity')

    ax2.plot(target[0,:,0], 'b', label='Target cos')
    ax2.plot(target[0,:,1], 'r', label='Target sin')
    ax2.plot(output[0,:,0], '--', color='deepskyblue', alpha=0.5, label='Output cos')
    ax2.plot(output[0,:,1], '--', color='lightcoral', alpha=0.5, label='Output sin')
    
    ax.set_ylabel("Angular velocity")
    ax.set_xlabel("Time")
    ax2.set_ylabel("Head direction")
    ax2.legend()

    return ax, ax2

def plot_losses_in_folder():
    main_exp_name='center_out/variable_N100_T250_Tr100/tanh/'
    folder = parent_dir+"/experiments/" + main_exp_name
    which = 'post'
    all_losses = np.empty((88,5000))
    all_losses[:] = np.nan
    fig, axs = plt.subplots(1, 2, figsize=(3, 3))
    for exp_i in range(88):
    #exp_i=10

        net, wi, wrec, wo, brec, h0, oth, training_kwargs, losses = load_all(main_exp_name, exp_i, which=which);
        all_losses[exp_i,:np.argmin(losses)] = losses[:np.argmin(losses)].copy()
        axs[0].plot(losses[:np.argmin(losses)].copy(), 'b',  alpha=.01); 
        #print(np.nanmin(losses), np.argmin(losses))
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

def plot_losses_and_trajectories(plot_losses_or_trajectories='losses', main_exp_name='angular_integration', model_name='low_gain', sparsity=.1, task=None):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    
    
    params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_name + '/param*.yml')[0]
    training_kwargs = yaml.safe_load(Path(params_path).read_text())
    exp_list = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/'+ model_name + "/result*")
    all_losses = np.zeros((len(exp_list), training_kwargs['n_epochs']))
    all_vallosses = np.zeros((len(exp_list), training_kwargs['n_epochs']))

    if not task:
        task = angularintegration_task(T=training_kwargs['T'], dt=training_kwargs['dt_task'], sparsity=sparsity)

    if plot_losses_or_trajectories:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax2 = ax.twinx()
    for exp_i, exp in enumerate(exp_list):
        # print(exp)
        
        with open(exp, 'rb') as handle:
            result = pickle.load(handle)
        
        # losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result
        losses = result[0]
        all_losses[exp_i, :len(losses)] = losses
        
        if plot_losses_or_trajectories == 'trajectories':
            try:
                losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result
            except:
                losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs, training_kwargs_ = result

            wi_init, wrec_init, wo_init, brec_init, h0_init = weights_last
            dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
            net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'],
                      nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
                      wi_init=wi_init, wrec_init=wrec_init, wo_init=wo_init, brec_init=brec_init, h0_init=h0_init, ML_RNN=training_kwargs['ml_rnn'])

            np.random.seed(12345)
            input, target, mask, output, loss = run_net(net, task, batch_size=1, return_dynamics=False, h_init=None)

            ax2.plot(output[0,:,0], '--', color='deepskyblue', alpha=0.5)
            ax2.plot(output[0,:,1], '--', color='lightcoral', alpha=0.5)
    
        elif plot_losses_or_trajectories == 'losses':
            ax.plot(losses, color='b', alpha=0.2, label='')
            # training_kwargs_ = result[-1]
            # print(training_kwargs_['g_in'], losses[-1])
            # print(result)
            if len(result)>7:
                # training_kwargs_['dataset_filename']
                validation_losses = result[1]
                all_vallosses[exp_i, :len(validation_losses)] = validation_losses
                # print(validation_losses)
                ax.plot(validation_losses, color='r', alpha=0.9)

    
    if plot_losses_or_trajectories == 'trajectories':
        ax, ax2 = plot_io(ax, ax2, input, output, target)
        # ax.set_ylim([0,2])
        # ax.set_ylim([0,2])

        plt.savefig(parent_dir+'/experiments/'+main_exp_name +'/'+ model_name + f'/trajectories{sparsity}.pdf')
        plt.show()
    elif plot_losses_or_trajectories == 'losses':
        mean = np.mean(all_losses, axis=0)
        var = np.var(all_losses, axis=0)
        
        ax.plot(range(mean.shape[0]), mean, label='Training loss')
        ax.plot(range(mean.shape[0]), np.mean(all_vallosses, axis=0),  label='Validation loss')

        # ax.errorbar(range(mean.shape[0]), mean, var, label='Training loss')
        # ax.errorbar(range(mean.shape[0]), np.mean(all_vallosses, axis=0), np.var(all_vallosses, axis=0), label='Validation loss')

        ax.set_yscale('log')
        ax.legend()
        plt.savefig(parent_dir+'/experiments/'+main_exp_name +'/'+ model_name + '/losses.pdf')
        plt.show()
        
    return all_losses

def plot_explained_variance(explained_variance): #np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(1, 1, figsize=(5, 3));
    ax.plot(np.arange(1,11), explained_variance, 'x')
    ax.plot([0,11], [.95,.95], 'r--')
    ax.set_xlim([.5,10.5])
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance")
    plt.savefig(inputdriven_folder_path+f'/explained_variance_{which}_{exp_i}.png', bbox_inches="tight", pad_inches=0.0)

def plot_losses_experiment(plot_losses_or_trajectories='losses', main_exp_name='angularintegration',
                           model_name='low_gain', training_label=None):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    colors = ['k', 'r',  'darkorange', 'g', 'b']

    
    params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_name + '/param*.yml')[0]
    training_kwargs = yaml.safe_load(Path(params_path).read_text())
    exp_list = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/'+ model_name + "/result*")
    all_losses = np.zeros((len(exp_list), training_kwargs['n_epochs']))
    all_vallosses = np.zeros((len(exp_list), training_kwargs['n_epochs']))


    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_lines = []
    parameters = []
    for exp_i, exp in enumerate(exp_list):
        
        
        with open(exp, 'rb') as handle:
            result = pickle.load(handle)
        training_kwargs = result[-1]
        parameters.append(training_kwargs[training_label])
        
        losses = result[0]
        all_losses[exp_i, :len(losses)] = losses
        
        l1, = ax.plot(losses, '', color=colors[exp_i])
        
        try:
            
            # training_kwargs_['dataset_filename']
            validation_losses = result[1]
            all_vallosses[exp_i, :len(validation_losses)] = validation_losses
            # print(validation_losses)
            l2, = ax.plot(validation_losses, '--', color=colors[exp_i])
        except:
            0
        plot_lines.append([l1, l2])


    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.set_yscale('log')
    # ax.legend(title=training_label.replace('_',' '))
    legend1 = plt.legend(plot_lines[0], ["Training", "Validation"], loc=5)
    ax.legend([l[0] for l in plot_lines], parameters)
    ax.add_artist(legend1)
    
    plt.savefig(parent_dir+'/experiments/'+main_exp_name +'/'+ model_name + '/losses.pdf')
    plt.show()
        
    return all_losses

def plot_all(model_names, main_exp_name='angularintegration'):
    print(parent_dir+'/experiments/' + main_exp_name +'/'+ model_names[0] + '/param*.yml')
    params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_names[0] + '/param*.yml')[0]
    # params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_name + '/param*.yml')[0]
    training_kwargs = yaml.safe_load(Path(params_path).read_text())

    mean_colors = ['k', 'r',  'darkorange', 'g', 'b']
    trial_colors = ['gray', 'lightcoral', 'moccasin', 'lime', 'deepskyblue']
    labels=['LSTM', r'$g=.5$', r'$g=1.5$', 'Orthogonal', 'QPTA']
    # plt.style.use('ggplot')
    plt.style.use('seaborn-dark-palette')


    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for model_i, model_name in enumerate(model_names):
        all_losses = plot_losses_and_trajectories(plot_losses_or_trajectories=None, main_exp_name=main_exp_name, model_name=model_name)
        for i in range(all_losses.shape[0]):
            ax.plot(all_losses[i,...], color=trial_colors[model_i], alpha=0.1)
        mean = np.mean(all_losses, axis=0)
        ax.plot(range(training_kwargs['n_epochs']), mean, color=mean_colors[model_i], label=labels[model_i])
        ax.set_yscale('log')
    ax.legend()
    plt.gca().set_ylim(top=1)
    plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/angular_losses.pdf')
    plt.show()
    
    
def plot_final_losses(main_exp_name, exp="triallength", mean_colors=['b'], trial_colors=['b'], labels=['']):
    sub_exp_names = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/*' + os.path.sep)
    
    if exp=="triallength":
        xlabel="Trial length"
        sub_exp_names = [float(PurePath(sub_exp_name).parts[-1]) for sub_exp_name in sub_exp_names]
    elif exp=="size":
        xlabel="Network size"
        sub_exp_names = [int(PurePath(sub_exp_name).parts[-1]) for sub_exp_name in sub_exp_names]
    sub_exp_names = sorted(sub_exp_names)
    print(sub_exp_names)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    xticks, positions = [], []
    for sub_exp_i, sub_exp_name in enumerate(sub_exp_names):
        print(sub_exp_name)
        sub_exp_name = str(sub_exp_name)
        for model_i, model_name in enumerate(model_names):
            print(model_name)
            print(parent_dir+'/experiments/' + main_exp_name +'/'+ sub_exp_name +'/'+ model_names[model_i])
            params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ sub_exp_name +'/'+ model_names[model_i] + '/param*.yml')[0]
            training_kwargs = yaml.safe_load(Path(params_path).read_text())
            all_losses = plot_losses_and_trajectories(plot_losses_or_trajectories=None, main_exp_name=main_exp_name+'/'+ sub_exp_name, model_name=model_name)
            final_losses = all_losses[:,-10:]
            final_losses = np.mean(final_losses, axis=1)
            final_losses_nonan = final_losses[~np.isnan(final_losses)]
            
            # ax.scatter([training_kwargs['T']/training_kwargs['dt_task']]*final_losses.shape[0], final_losses, marker='_', color=mean_colors[model_i], alpha=.9)
            # plt.bar(bins1[:-1], hist_data1, width=bar_width, alpha=0.5, label='Data 1')
            if exp=="triallength":
                bplot = ax.boxplot(final_losses_nonan, widths=25,
                                   positions=[np.log(training_kwargs['T']/training_kwargs['dt_task'])*sub_exp_i*50+35*model_i-70],
                                   patch_artist=True)
            
            elif exp=="size":
                bplot = ax.boxplot(final_losses_nonan, widths=10,
                                   positions=[sub_exp_i*100+15*model_i-35],
                                   patch_artist=True)

            bplot['boxes'][0].set_facecolor(mean_colors[model_i])

        
        if exp=="triallength":
            xticks.append(int(training_kwargs['T']/training_kwargs['dt_task']))
            positions.append(np.log(training_kwargs['T']/training_kwargs['dt_task'])*sub_exp_i*50)
        elif exp=="size":
            positions.append(sub_exp_i*100)
            xticks.append(int(training_kwargs['N_rec']))

    
    ax.set_xticks(np.array(positions),xticks)
    ax.set_xlim([positions[0]-100,positions[-1]+100])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Final loss")
    handles = [mpatches.Patch(color=color, label=f'Label {i}') for i, color in enumerate(mean_colors)]

    plt.legend(handles, labels)
    if exp=="triallength":
        plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/alltimes_finallosses.pdf', bbox_inches="tight")
    elif exp=="size":
        plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/allsizes_finallosses.pdf', bbox_inches="tight")

    plt.show()  
    
    
    
###network dynamics (trajectories)    
def plot_bestmodel_traj(model_names, main_exp_name='angularintegration', sparsity=1, i=0):
    params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_names[0] + '/param*.yml')[0]
    training_kwargs = yaml.safe_load(Path(params_path).read_text())
    
    mean_colors = ['k', 'r',  'darkorange', 'g', 'b']
    trial_colors = ['gray', 'lightcoral', 'moccasin', 'lime', 'deepskyblue']
    labels=['LSTM', r'$g=.5$', r'$g=1.5$', 'Orthogonal', 'QPTA']
    network_types = ['lstm_noforget', 'rnn', 'rnn', 'rnn', 'rnn']
    plt.style.use('seaborn-dark-palette')
    
    if len(model_names)==1:
        mean_colors=[mean_colors[i]]
        trial_colors=[trial_colors[i]]
        labels=[labels[i]]
        network_types = [network_types[i]]

    task = angularintegration_task(T=training_kwargs['T'], length_scale=training_kwargs['T']/12.8, dt=training_kwargs['dt_task'], sparsity=sparsity)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax2 = ax.twinx()
    for model_i, model_name in enumerate(model_names):
        params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_names[model_i] + '/param*.yml')[0]
        training_kwargs = yaml.safe_load(Path(params_path).read_text())
        training_kwargs['network_type'] = network_types[model_i]
        final_losses = plot_losses_and_trajectories(plot_losses_or_trajectories=None, main_exp_name=main_exp_name, model_name=model_name)[:,-1]
        minidx = np.nanargmin(final_losses)

        bestexp = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/'+ model_name + "/result*")[minidx]
        print(model_name, minidx, final_losses[minidx])
        with open(bestexp, 'rb') as handle:
            result = pickle.load(handle)

        if training_kwargs['network_type'] =='lstm_noforget':
            losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result
            w_init, u_init, bias_init, wo_init = weights_last

            dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
            net = LSTM_noforget(dims=dims,  readout_nonlinearity=training_kwargs['readout_nonlinearity'],
                      w_init=w_init, u_init=u_init, wo_init=wo_init, bias_init=bias_init)
            np.random.seed(123)
            input, target, mask, output, loss = run_lstm(net, task, batch_size=32, return_dynamics=False, init_states=None)
            
        else:
            losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result
            wi_init, wrec_init, wo_init, brec_init, h0_init = weights_last
            dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
            net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'],
                      nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
                      wi_init=wi_init, wrec_init=wrec_init, wo_init=wo_init, brec_init=brec_init, h0_init=h0_init, ML_RNN=training_kwargs['ml_rnn'])
            
            np.random.seed(123)
            input, target, mask, output, loss = run_net(net, task, batch_size=32, return_dynamics=False, h_init=None)

        ax2.plot(output[0,:,0], '--', color=mean_colors[model_i], label=labels[model_i])
        ax2.plot(output[0,:,1], '--', color=mean_colors[model_i])

    ax.plot(input[0,...], 'k', label='Angular valocity')
    ax.set_ylabel("Angular velocity")
    ax2.plot(target[0,:,0], 'b')
    ax2.plot(target[0,:,1], 'r')
    plt.xlabel("Time")
    ax2.set_ylabel("Head direction (sin, cos)")
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=6)
    plt.savefig(parent_dir+'/experiments/'+main_exp_name+f'/trajectories{sparsity}.pdf', bbox_inches="tight")
    plt.show()
    
def load_best_model_from_grid(experiment_folder, model_name, task):
    # df = pd.read_csv(parent_dir+'/experiments/'+experiment_folder+'/grid_search_'+model_name+'.csv', sep=',')
    
    df = pd.read_pickle(parent_dir+'/experiments/'+experiment_folder+'/grid_search_'+model_name+'.pickle')
    
    losses = np.vstack(df['losses'])

    idx_min_meanloss = np.argmin(np.mean(losses, axis=1))
    idx_min_finalloss = df['final_loss'].idxmin()

    plt.plot(losses[np.argmin(np.mean(losses, axis=1))], 'b');
    plt.plot(losses[df['final_loss'].idxmin()], 'r');
    plt.plot(losses.T, 'k', alpha=.1);
    plt.yscale('log')
    plt.show()
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax2 = ax.twinx()
    weights = df['weights_last'][idx_min_finalloss]
    net = load_net_from_weights(weights, dt=.1)
    plot_io_net(net, task, ax=ax, ax2=ax2)
    
    weights = df['weights_last'][idx_min_meanloss]
    net = load_net_from_weights(weights, dt=.1)
    plot_io_net(net, task, ax=ax, ax2=ax2)
    
    df_final = df.drop(['losses', 'weights_last'], axis=1)
    df_final = df_final.fillna("None")
    
    param_keys = df_final.columns
    param_keys = param_keys.drop(['trial', 'final_loss'])
    param_keys=param_keys.tolist()
    min_final_loss = df.iloc[df['final_loss'].idxmin()]
    print(df['final_loss'].min())
    print(min_final_loss)
    min_final_meanloss = df_final.groupby(param_keys).mean()['final_loss'].idxmin()
    
    print(min_final_meanloss)
    # print(min_final_meanloss)
    # print("LLL", len(min_final_loss['weights_last']))
    # wi_init, wrec_init, wo_init, brec_init, h0_init = min_final_loss['weights_last']
    # dims = (1, 200, 2)
    # net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'],
    #           nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
    #           wi_init=wi_init, wrec_init=wrec_init, wo_init=wo_init, brec_init=brec_init, h0_init=h0_init, ML_RNN=training_kwargs['ml_rnn'])

    # np.random.seed(12345)
    # input, target, mask, output, loss = run_net(net, task, batch_size=32, return_dynamics=False, h_init=None)

    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax2 = ax.twinx()
    # ax2.plot(output[0,:,0], '--', color='deepskyblue', alpha=0.5)
    # ax2.plot(output[0,:,1], '--', color='lightcoral', alpha=0.5)
    # ax, ax2 = plot_io(ax, ax2, input, output, target)
    # # plt.savefig(parent_dir+'/experiments/'+main_exp_name +'/'+ model_name + '/trajectories{sparsity}.pdf')
    # plt.show()
    
    return df



def plot_all_trajs_model(main_exp_name, model_name, T=128, which='post', hidden_i=0, input_length=10,
                         timepart='all', num_of_inputs=51, plot_from_to=(0,None), pca_from_to=(0,None),
                         plot_output=False, input_range=(-3,3)):
    pca_before_t, pca_after_t = pca_from_to
    before_t, after_t = plot_from_to
    
    exp_list = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/'+ model_name + "/result*")
    
    norm = mplcolors.Normalize(vmin=-.5,vmax=.5)
    norm = norm(np.linspace(-.5, .5, num=num_of_inputs, endpoint=True))
    cmap = cmx.get_cmap("coolwarm")
    
    cmap2 = plt.get_cmap('hsv')
    norm2 = mpl.colors.Normalize(-np.pi, np.pi)
    
    # fig, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)
    fig, axes = plt.subplots(3, 6, figsize=(18, 9), sharex=False, sharey=False)
    axes1 = axes[:,:3].flatten()
    axes2 = axes[:,3:].flatten()

    for exp_i, exp in enumerate(exp_list[:9]):
        trajectories, start, target, output = get_hidden_trajs(main_exp_name, model_name, exp, T=T, which=which, input_length=input_length,
                                                               timepart=timepart, num_of_inputs=num_of_inputs, pca_from_to=pca_from_to, input_range=input_range)

        for trial_i in range(trajectories.shape[0]):
            axes1[exp_i].plot(trajectories[trial_i,before_t:after_t,0], trajectories[trial_i,before_t:after_t,1], '-',
                              color=cmap(norm[trial_i]))
            if np.linalg.norm(trajectories[trial_i,-2,:]-trajectories[trial_i,-1,:])  < 1e-4:
                axes1[exp_i].scatter(trajectories[trial_i,-1,0], trajectories[trial_i,-1,1], marker='.', s=100, color=cmap(norm[trial_i]), zorder=100)

        axes1[exp_i].set_axis_off()
        axes1[exp_i].scatter(start[0], start[1], marker='.', s=100, color='k', zorder=100)
        
        if plot_output:
            x = np.linspace(0, 2*np.pi, 1000)
            axes2[exp_i].plot(np.cos(x), np.sin(x), 'k', alpha=.5, linewidth=5, zorder=-1)
            for trial_i in range(output.shape[0]):
                # if trial_i<output.shape[0]-1:
                #     axes2[exp_i].plot([target[trial_i,-1,0], target[trial_i+1,-1,0]], [target[trial_i,-1,1], target[trial_i+1,-1,1]], '--', color=cmap(norm[trial_i]), alpha=.5)
                # axes2[exp_i].scatter(target[trial_i,-1,0], target[trial_i,-1,1], color=cmap(norm[trial_i]), alpha=.1)
                # axes2[exp_i].plot(output[trial_i,:,0], output[trial_i,:,1], '-', color=cmap(norm[trial_i]))
                if np.linalg.norm(trajectories[trial_i,-2,:]-trajectories[trial_i,-1,:])  < 1e-4:
                    axes2[exp_i].scatter(output[trial_i,-1,0], output[trial_i,-1,1], marker='.', s=100, color=cmap(norm[trial_i]), zorder=100)
            
            for t in range(output.shape[1]):
                axes2[exp_i].plot([target[trial_i,t,0], output[trial_i,t,0]],
                                 [target[trial_i,t,1], output[trial_i,t,1]], '-', 
                                 color=cmap2(norm2(np.arctan2(target[trial_i,t,1], target[trial_i,t,0]))))
                # ax.scatter(xval, yval, c=xval, s=300, cmap=colormap, norm=norm, linewidths=0)

            axes2[exp_i].set_axis_off()
            
    if not timepart:
        plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/trajpca_{which}_{after_t}to{before_t}.png', bbox_inches="tight")
        plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/trajpca_{which}_{after_t}to{before_t}.pdf', bbox_inches="tight")
    else:
        plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/trajpca_{which}_{timepart}.png', bbox_inches="tight")
        plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/trajpca_{which}_{timepart}.pdf', bbox_inches="tight")
    
    if plot_output:
        plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/output_{which}__{after_t}to{before_t}.png', bbox_inches="tight")
        plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/output_{which}__{after_t}to{before_t}.pdf', bbox_inches="tight")
            
            

        
def get_traininginfo_exp(params_folder, exp_i=0, which='post'):
    exp_list = glob.glob(params_folder + "/result*")
    exp = exp_list[exp_i]
    params_path = glob.glob(params_folder + '/param*.yml')[0]
    training_kwargs = yaml.safe_load(Path(params_path).read_text())
    with open(exp, 'rb') as handle:
        result = pickle.load(handle)
    
    try:
        losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs, training_kwargs = result
        
        
    except:
        losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result


    return losses, gradient_norms, epochs, rec_epochs, weights_train
    



def plot_input_driven_trajectory_2d(traj, traj_pca, wo,
                                    plot_asymp=False, limcyctol=1e-2, mindtol=1e-4,
                                    fxd_points = None, ops_fxd_points=None, 
                                    h_stabilities=None, o_stabilities=None, 
                                    ax=None, plot_epoch=False):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    
    norm = mplcolors.Normalize(vmin=-.5,vmax=.5)
    num_of_inputs = traj.shape[0]
    norm = norm(np.linspace(-.5, .5, num=num_of_inputs, endpoint=True))
    cmap = cmx.get_cmap("coolwarm")
    norm2 = mpl.colors.Normalize(-np.pi, np.pi)
    cmap2 = plt.get_cmap('hsv')
    stab_colors = np.array(['g', 'pink', 'red'])
    
    output = np.dot(traj, wo)
    for trial_i in range(traj.shape[0]):
        ax.plot(traj_pca[trial_i,:,0], traj_pca[trial_i,:,1], color=cmap(norm[trial_i]))
        if plot_asymp:
            target_angle = np.arctan2(output[trial_i,-1,1], output[trial_i,-1,0])
            idx, mind = identify_limit_cycle(traj[trial_i,:,:], tol=limcyctol) #find recurrence
            if idx:
                ax.plot(traj_pca[trial_i,idx:,0], traj_pca[trial_i,idx:,1], '-', linewidth=3, color=cmap2(norm2(target_angle)), zorder=100)
                
                if mind<mindtol:
                    ax.scatter(traj_pca[trial_i,-1,0], traj_pca[trial_i,-1,1], marker='.', s=100, color=cmap2(norm2(target_angle)), zorder=110)
    
    ax.set_axis_off()
    if np.any(fxd_points):
        pca_fxd_points = pca.transform(fxd_points)
        ax.scatter(pca_fxd_points[:,0], pca_fxd_points[:,1], marker='s',
                   c=stab_colors[h_stabilities], alpha=.5, zorder=101)
    if np.any(ops_fxd_points):
        pca_ops_fxd_points = pca.transform(ops_fxd_points)
        if o_stabilities==None:
            ax.scatter(pca_ops_fxd_points[:,0], pca_ops_fxd_points[:,1],
                       marker='x', c='k', alpha=.5, zorder=101)
        else:
            ax.scatter(pca_ops_fxd_points[:,0], pca_ops_fxd_points[:,1],
                   marker='x', c=stab_colors[o_stabilities], alpha=.5, zorder=101)
            
    if plot_epoch:
        plt.text(.0, 0., s=plot_epoch, fontsize=10)
        
        
        
########Lyapunov exponents
def calc_LEs(net, T=10, from_t_step=0):
    task = angularintegration_task(T=T, dt=.1, sparsity=0); #just creates zero input batches
    input, target, mask, output, loss, trajectories = run_net(net, task, batch_size=1, return_dynamics=True, h_init=None);
    lyap_spec, lyaps = calculate_lyapunov_spectrum(act_fun=tanh_jacobian, W=net.wrec.cpu().detach().numpy(), b=net.brec.cpu().detach().numpy(),
                                                   tau=1, x_solved=trajectories[0,...], delta_t=net.dt, from_t_step=from_t_step)

    return sorted(lyap_spec, reverse=True), trajectories

def plot_allLEs_model(main_exp_name, model_name, which='last', T=10, from_t_step=0, mean_color='b', trial_color='b', label='', ax=None, save=False, hidden_i=0):
    print(parent_dir+'/experiments/' + main_exp_name +'/'+ model_name + '/param*.yml')
    params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_name + '/param*.yml')[0]
    training_kwargs = yaml.safe_load(Path(params_path).read_text())
    exp_list = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/'+ model_name + "/result*")
    
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    lyap_specs = np.zeros((len(exp_list), training_kwargs['N_rec']))
    for exp_i, exp in enumerate(exp_list):
        print(exp)
        with open(exp, 'rb') as handle:
            result = pickle.load(handle)
            
        losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result
        if which=='post':
            wi_init, wrec_init, wo_init, brec_init, h0_init = weights_last
        elif which=='pre':
            wi_init, wrec_init, wo_init, brec_init, h0_init = weights_init
        dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
        net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'],
                  nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
                  wi_init=wi_init, wrec_init=wrec_init, wo_init=wo_init, brec_init=brec_init, h0_init=h0_init, ML_RNN=training_kwargs['ml_rnn'])
        
        lyap_spec, trajectories = calc_LEs(net, T=T, from_t_step=from_t_step)
        lyap_specs[exp_i, :] = lyap_spec
        ax.plot(lyap_spec, color=trial_color, alpha=.1)
    
    ax.plot(np.mean(lyap_specs, axis=0), color=mean_color, label=label)
    # plt.show()
    # if save:
    #     plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/les_{which}.pdf', bbox_inches="tight")
    #     plt.close()
    return ax


def plot_allLEs(main_exp_name, mean_colors, trial_colors, labels, T=10, from_t_step=0, which='post'):
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    for model_i, model_name in enumerate(model_names):
        model_i += 1
        plot_allLEs_model(main_exp_name, model_name, first_or_last=which, from_t_step=from_t_step, T=T,
                          ax=ax, mean_color=mean_colors[model_i], trial_color=trial_colors[model_i], label=labels[model_i])
    ax.set_ylabel("Lyapunov exponent")

    plt.legend()
    plt.savefig(parent_dir+'/experiments/'+main_exp_name+f'/les_{which}.pdf', bbox_inches="tight")
    plt.show()




###plot recurrent sets
def plot_recs_ring_3d(recurrences, recurrences_pca, cmap, norm, ax=None):
    
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        
    for r_i, recurrence in enumerate(recurrences):
        recurrence_pca = np.array(recurrences_pca[r_i]).T
    
        if np.array(recurrences_pca[r_i]).shape[0]>1:
            recurrence_pca = recurrences_pca[r_i][:,:3]
            output = np.dot(recurrences[r_i], wo)
            output_angle = np.arctan2(output[...,1], output[...,0])
    
            segments = np.stack([recurrence_pca[:-1], recurrence_pca[1:]], axis=1)
            lc = Line3DCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(output_angle)
    
            # Plot the line segments in 3D
            ax.add_collection3d(lc)
    
        else:
        # for t, point in enumerate(recurrence):
            point = recurrence[-1]
            # print("P", point)
            output = np.dot(point, wo)
            output_angle = np.arctan2(output[...,1], output[...,0])
    
            point_pca = recurrence_pca
            ax.scatter(point_pca[0], point_pca[1], point_pca[2],
                        marker='.', s=100, color=cmap(norm(output_angle)), zorder=1000)
            
    return ax


def plot_points_ring_3d(points, wo, pca, marker_style, markercolor='r', ax=None, zorder=0):
    cmap = plt.get_cmap('hsv')
    norm = mpl.colors.Normalize(-np.pi, np.pi)

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    for p_i, point in enumerate(points):
        output = np.dot(point, wo)
        output_angle = np.arctan2(output[...,1], output[...,0])
        point_pca = pca.transform(point.reshape((1,-1)))
        if markercolor=='ring':
            color=cmap(norm(output_angle))
        else:
            color=markercolor
        if marker_style == "full":
            ax.scatter(point_pca[0][0], point_pca[0][1], point_pca[0][2],
                    s=25, zorder=zorder, facecolors=color, edgecolors=color)
        else:
            facecolors = 'w'
            ax.scatter(point_pca[0][0], point_pca[0][1], point_pca[0][2],
                    s=25, zorder=zorder, facecolors=facecolors, edgecolors=color)
        
        
        
def plot_binned_ring_3d(all_bin_locs, ax=None):
    
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        
    ax.scatter(all_bin_locs[:,0], all_bin_locs[:,1], all_bin_locs[:,2],
                marker='.', s=.1, color='k', zorder=-100, alpha=.9)                 
    ax = set_3daxes(ax)
    return ax

def plot_recs_3d_ring(recurrences, recurrences_pca, wo, cmap, norm, ax=None):
    
    
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    
    for r_i, recurrence in enumerate(recurrences):
        recurrence_pca = np.array(recurrences_pca[r_i]).T
        
        if np.array(recurrences_pca[r_i]).shape[0]>1:
            recurrence_pca = recurrences_pca[r_i][:,:3]
            output = np.dot(recurrences[r_i], wo)
            output_angle = np.arctan2(output[...,1], output[...,0])
            
            segments = np.stack([recurrence_pca[:-1], recurrence_pca[1:]], axis=1)
            lc = Line3DCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(output_angle)
            
            # Plot the line segments in 3D
            ax.add_collection3d(lc)
        
        else:
        # for t, point in enumerate(recurrence):
            point = recurrence[-1]
            # print("P", point)
            output = np.dot(point, wo)
            output_angle = np.arctan2(output[...,1], output[...,0])
            
            point_pca = recurrence_pca
            ax.scatter(point_pca[0], point_pca[1], point_pca[2],
                        marker='.', s=100, color=cmap(norm(output_angle)), zorder=1000)
    
    return ax


##plot slow manifold
def plot_slow_manifold_ring_3d(saddles, fxd_pnts, wo, pca, exp_name,
                               all_bin_locs_pca=None,
                               recurrences=[], recurrences_pca=[],
                               trajectories=[], traj_alpha=.1, proj_mult_val=1.5,
                               proj_2d_color='lightgrey', figname_postfix=''):
    assert all_bin_locs_pca is not None or np.any(trajectories), "provide all_bin_locs_pca or trajectories"
    cmap = plt.get_cmap('hsv')
    norm = mpl.colors.Normalize(-np.pi, np.pi)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.view_init(elev=45., azim=45)
    plot_points_ring_3d(fxd_pnts, wo, pca, marker_style='full', markercolor='ring', ax=ax, zorder=100)
    plot_points_ring_3d(saddles, wo, pca, marker_style='', markercolor='ring', ax=ax, zorder=50)
    plot_recs_3d_ring(recurrences, recurrences_pca, wo, cmap, norm, ax=ax)  

    for traj in trajectories:
        ax.plot(traj[:,0], traj[:,1], traj[:,2], 'k', alpha=traj_alpha, zorder=0)
    
    if np.any(all_bin_locs_pca!=None):
        ax.scatter(all_bin_locs_pca[:,0], all_bin_locs_pca[:,1], all_bin_locs_pca[:,2],
                    marker='.', s=.1, color='k', zorder=-100, alpha=.9)
        
        ax.scatter(proj_mult_val*np.min(all_bin_locs_pca[:,0])*np.ones_like(all_bin_locs_pca[:,2]), all_bin_locs_pca[:,1], all_bin_locs_pca[:,2],
                      marker='.', s=.1, color=proj_2d_color, zorder=-200, alpha=.9)
        ax.scatter(all_bin_locs_pca[:,0], proj_mult_val*np.min(all_bin_locs_pca[:,1])*np.ones_like(all_bin_locs_pca[:,2]), all_bin_locs_pca[:,2],
                      marker='.', s=.1, color=proj_2d_color, zorder=-200, alpha=.9)
        ax.scatter(all_bin_locs_pca[:,0], all_bin_locs_pca[:,1], proj_mult_val*np.min(all_bin_locs_pca[:,2])*np.ones_like(all_bin_locs_pca[:,2]),
                      marker='*', s=.1, color=proj_2d_color, zorder=-200, alpha=.9)
    
    if np.any(trajectories):
        for traj in trajectories:
            ax.plot(proj_mult_val*np.min(trajectories[:,:,0])*np.ones_like(traj[:,0]), traj[:,1], traj[:,2], 'k', alpha=0.1, zorder=-2000)

            ax.plot(traj[:,0], proj_mult_val*np.min(trajectories[:,:,1])*np.ones_like(traj[:,1]), traj[:,2], 'k', alpha=0.1, zorder=-2000)

            ax.plot(traj[:,0], traj[:,1], proj_mult_val*np.min(trajectories[:,:,2])*np.ones_like(traj[:,2]), 'k', alpha=0.1, zorder=-2000)
    
    for axis in range(3):
        if np.any(trajectories):
            value = proj_mult_val*np.min(trajectories[:,:,axis])

        else:
            value = proj_mult_val*np.min(all_bin_locs_pca[:,axis])

        scatter_projection_3d(fxd_pnts, wo, pca, axis, value, cmap, norm, marker_style='full', markercolor=proj_2d_color, ax=ax)
        scatter_projection_3d(saddles, wo, pca, axis, value, cmap, norm, marker_style='full', markercolor=proj_2d_color, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3", rotation=90); ax.zaxis.labelpad=-15
    plt.savefig(parent_dir+'/experiments/'+exp_name+f'/slow_manifold_3d2d_{figname_postfix}.pdf', bbox_inches="tight")
    # ax = set_3daxes(ax)         


def video_slow_manifold_ring_3d(saddles, fxd_pnts, all_bin_locs_pca, wo, pca, exp_name,
                               proj_mult_val, 
                               recurrences=[], recurrences_pca=[],
                               trajectories=[],
                               proj_2d_color='lightgrey', figname_postfix=''):
    cmap = plt.get_cmap('hsv')
    norm = mpl.colors.Normalize(-np.pi, np.pi)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.view_init(elev=45., azim=45)
    plot_points_ring_3d(fxd_pnts, wo, pca, marker_style='full', markercolor='ring', ax=ax, zorder=100)
    plot_points_ring_3d(saddles, wo, pca, marker_style='', markercolor='ring', ax=ax, zorder=50)
    plot_recs_3d_ring(recurrences, recurrences_pca, wo, cmap, norm, ax=ax)  

    for traj in trajectories:
        ax.plot(traj[:,0], traj[:,1], traj[:,2], 'k', alpha=0.1, zorder=-1000)
    
    for axis in range(3):
        value = proj_mult_val*np.min(all_bin_locs_pca[:,axis])
        scatter_projection_3d(fxd_pnts, wo, pca, axis, value, cmap, norm, marker_style='full', markercolor=proj_2d_color, ax=ax)
        scatter_projection_3d(saddles, wo, pca, axis, value, cmap, norm, marker_style='full', markercolor=proj_2d_color, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3", rotation=90); ax.zaxis.labelpad=-15
    plt.savefig(parent_dir+'/experiments/'+exp_name+f'/slow_manifold_3d2d_{figname_postfix}.pdf', bbox_inches="tight")



def plot_slowpoints(params_folder, exp_i):
    params_folder = parent_dir+'/experiments/' + main_exp_name +'/'+ model_name
    wi, wrec, wo, brec, h0, training_kwargs = get_params_exp(params_folder, exp_i=exp_i)
    trajectories, traj_pca, start, target, output, input_proj, pca, explained_variance = get_hidden_trajs(wi, wrec, wo, brec, h0, training_kwargs,
                                                                                      T=2*input_length, input_length=input_length, which=which,
                                                                                      pca_from_to=pca_from_to,
                                                                                      num_of_inputs=num_of_inputs, input_range=input_range)
    fig, ax = plt.subplots(1, 1, figsize=(3, 3)); 
    # # plot_average_vf(trajectories, wi, wrec, brec, wo, input_length=input_length, 
    # #                 num_of_inputs=2, input_range=(-.51,-.5), ax=ax)
    # # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/vf_{which}.pdf', bbox_inches="tight")
    traj = trajectories[:10,::50,:].reshape((-1,training_kwargs['N_rec']))
    fxd_points, speeds = find_slow_points(wrec, brec, dt=training_kwargs['dt_rnn'], trajectory=traj)
    # #pca projection
    # proj_fxd_points = pca.transform(fxd_points.reshape(-1, training_kwargs['N_rec']))
    # proj_h0 = pca.transform(h0.reshape(1, -1))
    # #output projection
    proj_fxd_points = np.dot(fxd_points, wo)

    proj_h0 = np.dot(h0, wo)
    ax.scatter(proj_fxd_points[:,0], proj_fxd_points[:,1])    
    # ax.scatter(proj_trajectory[:,0], proj_trajectory[:,1])    

    ax.scatter(proj_h0[0], proj_h0[1], c='k')
    
    cmap2 = plt.get_cmap('hsv')
    norm2 = mpl.colors.Normalize(-np.pi, np.pi)
    x = np.linspace(-np.pi, np.pi, 1000)
    ax.scatter(np.cos(x), np.sin(x), color=cmap2(norm2(x)), alpha=.5, s=2, zorder=-1)



####vector field
def plot_vf_on_ring_spline(wo, cs, npoints, fxd_pnt_thetas=[], stabilities=[]):
    
    xs = np.arange(-np.pi, np.pi, 2*np.pi/npoints)
    csx=cs(xs);
    
    fig, ax = plt.subplots(1, 1, figsize=(3, 3)); 
    for i in range(csx.shape[0]):
        x,y=np.cos(xs[i]),np.sin(xs[i])
        x,y=np.dot(wo.T,csx[i])
        u,v=np.dot(wo.T, tanh_ode(0, csx[i], wrec, brec, tau=10))
        plt.quiver(x,y,u,v)
    if np.any(fxd_pnt_thetas):
        for i,theta in enumerate(fxd_pnt_thetas):
            x,y=np.cos(theta),np.sin(theta)
            if stabilities[i]==1:
                plt.plot(x,y,'.g')
            else:
                plt.plot(x,y,'.r')
    plt.axis('off'); 
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None);
    plt.savefig(parent_dir+'/experiments/'+main_exp_name+f'/vf_on_ring_exp{exp_i}.pdf', bbox_inches="tight")

def plot_two_rings(net, batch_size=256):    
    # main_exp_name='center_out/variable_N200_T500/tanh/'
    # exp_i=0
    # net, wi, wrec, wo, brec, h0, oth, training_kwargs = load_all(main_exp_name, exp_i, which='post'); 

    T=1e5; dt=.1; from_t=300; task = center_out_reaching_task(T=T, dt=dt, time_until_cue_range=[250, 251], angles_random=False);
    input, target, mask, output, trajectories = simulate_rnn(net, task, T, batch_size)
    T=int(1e5); dt=.1; from_t=300; task = center_out_reaching_task(T=T, dt=dt, time_until_cue_range=[T, T+1], angles_random=False);
    input_w, target_w, mask_w, output_w, trajectories_w = simulate_rnn(net, task, T, batch_size)
    n_rec = net.dims[1]
    n_components=10
    pca = PCA(n_components=n_components)
    invariant_manifold = trajectories[:,:,:].reshape((-1,n_rec))
    pca.fit(invariant_manifold)

    from_t = 150
    to_t = 300
    invariant_manifold = trajectories_w[:,from_t:to_t,:].reshape((-1,n_rec))
    traj_pca_flat_1st = pca.transform(invariant_manifold).reshape((-1,n_components))
    traj_pca_1st = pca.transform(invariant_manifold).reshape((batch_size,-1,n_components))

    from_t = 300
    to_t = 1500
    invariant_manifold = trajectories_w[:,from_t:to_t,:].reshape((-1,n_rec))
    traj_pca_flat_conn = pca.transform(invariant_manifold).reshape((-1,n_components))
    traj_pca_conn = pca.transform(invariant_manifold).reshape((batch_size,-1,n_components))

    from_t = 1000
    to_t = 1200
    invariant_manifold = trajectories[:,from_t:to_t,:].reshape((-1,n_rec))
    traj_pca_flat_2nd = pca.transform(invariant_manifold).reshape((-1,n_components))
    traj_pca_2nd = pca.transform(invariant_manifold).reshape((batch_size,-1,n_components))

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.view_init(elev=-90., azim=0, roll=0)
    for i in range(batch_size):
        ax.plot(traj_pca_1st[i,:,0], traj_pca_1st[i,:,1], traj_pca_1st[i,:,2],
                    marker='.', color='r', zorder=100, alpha=.9);
        ax.plot(traj_pca_conn[i,:,0], traj_pca_conn[i,:,1], traj_pca_conn[i,:,2],
                    marker='.', color='k', zorder=0, alpha=.9);
        ax.plot(traj_pca_2nd[i,:,0], traj_pca_2nd[i,:,1], traj_pca_2nd[i,:,2],
                    marker='.', color='b', zorder=-100, alpha=.9);
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3", rotation=90); ax.zaxis.labelpad=-15
    plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/2rings_3d_top.pdf', bbox_inches="tight")


def plot_vf_diff_ring(xs, csx, diff, fxd_pnt_thetas, stabilities):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=45., azim=45, roll=0)
    x,y=np.dot(csx, wo).T
    x=np.cos(xs)
    y=np.sin(xs)
    ax.plot(x, y, diff)
    ax.set_xticks([-1,1])
    ax.set_yticks([-1,1])
    for i,theta in enumerate(fxd_pnt_thetas):
        x,y=np.cos(theta+np.pi/50),np.sin(theta+np.pi/50)
        if stabilities[i]==1:
            plt.plot(x,y,0,'.g')
        else:
            plt.plot(x,y,0,'.r')
    #fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None);
    plt.ticklabel_format(style='sci', axis='z', scilimits=(0,0))
    ax.set_zlabel("", rotation=90); ax.zaxis.labelpad=-15
    plt.savefig(parent_dir+'/experiments/'+main_exp_name+f'/difference_on_ring_closest_exp{exp_i}.pdf', bbox_inches="tight")
    
def scatter_projection_3d(points, wo, pca, axis, value, cmap, norm, marker_style, markercolor='r', ax=None):
    
    if marker_style == "full":
        facecolors = None
    else:
        facecolors = 'w'
    
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    for p_i, point in enumerate(points):
        output = np.dot(point, wo)
        output_angle = np.arctan2(output[...,1], output[...,0])
        point_pca = pca.transform(point.reshape((1,-1)))
        
        if markercolor=='ring':
            color=cmap(norm(output_angle))
        else:
            color=markercolor

        if axis==0:
            ax.scatter(value, point_pca[0][1], point_pca[0][2],
                    s=25, zorder=-150, color=color, facecolors=facecolors, edgecolors=color)
            
        elif axis==1:
            ax.scatter(point_pca[0][0], value, point_pca[0][2],
                    s=25, zorder=-150, color=color, facecolors=facecolors, edgecolors=color)
        else:
            ax.scatter(point_pca[0][0], point_pca[0][1], value,
                    s=25, zorder=-150, color=color, facecolors=facecolors, edgecolors=color)
    
        
def plot_inputdriven_trajectory_3d(traj, input_length, plot_traj=True,
                                    recurrences=None, recurrences_pca=None, wo=None,
                                    fxd_points=None, ops_fxd_points=None,
                                    h_stabilities=None,
                                    elev=20., azim=-35, roll=0,
                                    lims=[], plot_epoch=False,
                                    cmap = cmx.get_cmap("coolwarm")):
    #needs pca
    
    num_of_inputs = traj.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    norm = mplcolors.Normalize(vmin=-.5,vmax=.5)
    norm = norm(np.linspace(-.5, .5, num=num_of_inputs, endpoint=True))

    norm2 = mpl.colors.Normalize(-np.pi, np.pi)
    cmap2 = plt.get_cmap('hsv')

    stab_colors = np.array(['g', 'pink', 'red'])
    
    if plot_traj:
        for trial_i in range(traj.shape[0]):
            ax.plot(traj[trial_i,:input_length,0], traj[trial_i,:input_length,1], zs=traj[trial_i,:input_length,2],
                    zdir='z', color=cmap(norm[trial_i]))
            ax.plot(traj[trial_i,input_length:,0], traj[trial_i,input_length:,1], zs=traj[trial_i,input_length:,2],
                    linestyle='--', zdir='z', color=cmap(norm[trial_i]))

    if recurrences is not None:

        for r_i, recurrence in enumerate(recurrences):
            recurrence_pca = np.array(recurrences_pca[r_i]).T

            if np.array(recurrences_pca[r_i]).shape[0]>1:
                recurrence_pca = recurrences_pca[r_i][:,:3]
                output = np.dot(recurrences[r_i], wo)
                output_angle = np.arctan2(output[...,1], output[...,0])
                
                segments = np.stack([recurrence_pca[:-1], recurrence_pca[1:]], axis=1)
                lc = Line3DCollection(segments, cmap=cmap2, norm=norm2)
                lc.set_array(output_angle)
                
                # Plot the line segments in 3D
                ax.add_collection3d(lc)

            else:
            # for t, point in enumerate(recurrence):
                point = recurrence[-1]
                # print("P", point)
                output = np.dot(point, wo)
                output_angle = np.arctan2(output[...,1], output[...,0])

                point_pca = recurrence_pca
                ax.scatter(point_pca[0], point_pca[1], point_pca[2],
                            marker='.', s=100, color=cmap2(norm2(output_angle)), zorder=100)


    if np.any(fxd_points):
        pca_fxd_points = pca.transform(fxd_points)
        if h_stabilities==None:
            ax.scatter(pca_fxd_points[:,0], pca_fxd_points[:,1], pca_fxd_points[:,2],
                       marker='s', color='k',     alpha=.5, zorder=101)
        else:
            ax.scatter(pca_fxd_points[:,0], pca_fxd_points[:,1], pca_fxd_points[:,2],
                       marker='s', color=stab_colors[h_stabilities],     alpha=.5, zorder=101)
    if np.any(ops_fxd_points):
        pca_ops_fxd_points = pca.transform(ops_fxd_points)
        ax.scatter(pca_ops_fxd_points[:,0], pca_ops_fxd_points[:,1], pca_ops_fxd_points[:,2],
                   marker='x', color='k', alpha=.5, zorder=101)
        
    # if plot_epoch:
    #     plt.text(.0, 0., 0, s=plot_epoch, fontsize=10)

    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if lims!=[]:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        ax.set_zlim(lims[2])
    # ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.set_axis_off()
    ax.plot([-10, 10], [0,0], zs=[0,0], color='k', zorder=-100)
    ax.plot( [0,0], [-10, 10], zs= [0,0], color='k', zorder=-100)
    ax.plot( [0,0], [0,0], zs=[-10, 10], color='k', zorder=-100)
    ax = set_3daxes(ax)

    
def plot_output_trajectory(traj, wo, input_length, plot_traj=True,
                           fxd_points=None, ops_fxd_points=None,
                           h_stabilities=None, o_stabilities=None,
                           plot_asymp=False, limcyctol=1e-2, mindtol=1e-4, ax=None,
                           xylims=[-1.2,1.2], plot_epoch=False,
                           num_of_inputs = 0,
                           cmap = cmx.get_cmap("coolwarm")):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    num_of_inputs = traj.shape[0]
    norm = mplcolors.Normalize(vmin=-.5,vmax=.5)
    norm = norm(np.linspace(-.5, .5, num=num_of_inputs, endpoint=True))

    norm2 = mpl.colors.Normalize(-np.pi, np.pi)
    cmap2 = plt.get_cmap('hsv')
    norm3 = Normalize(-np.pi, np.pi)

    stab_colors = np.array(['g', 'pink', 'red'])
    x = np.linspace(-np.pi, np.pi, 1000)
    # axes[1].plot(np.cos(x), np.sin(x), 'k', alpha=.5, linewidth=5, zorder=-1)
    ax.scatter(np.cos(x), np.sin(x), color=cmap2(norm2(x)), alpha=.1, s=5, zorder=100) #ring
    
    output = np.dot(traj, wo)
    for trial_i in range(traj.shape[0]):
        if plot_traj:
            # ax.plot(output[trial_i,0,0], output[trial_i,0,1], '.', c='k', zorder=100) #starting point
            ax.plot(output[trial_i,:input_length,0], output[trial_i,:input_length,1], '-', color=cmap(norm[trial_i]), alpha=.5) #all trajectories
            ax.plot(output[trial_i,input_length:,0], output[trial_i,input_length:,1], '--', color=cmap(norm[trial_i]), alpha=.5) #all trajectories
    
        if plot_asymp:
            target_angle = np.arctan2(output[trial_i,-1,1], output[trial_i,-1,0])
            idx, mind = identify_limit_cycle(traj[trial_i,:,:], tol=limcyctol) #find recurrence
            if idx and not mind<mindtol:
                # ax.plot(output[trial_i,idx:,0], output[trial_i,idx:,1], '-', linewidth=3, color=cmap2(norm2(target_angle)), zorder=100)
                out_i = output[trial_i,idx:,:]
                out_i = out_i.reshape(-1, 1, 2)
                # print(out_i[:-1].shape)
                output_angle = np.arctan2(out_i[:,0,1], out_i[:,0,0])
                segments = np.concatenate([out_i[:-1], out_i[1:]], axis=1)
                lc = LineCollection(segments, cmap=cmap2, norm=norm3)
                lc.set_array(output_angle)
                ax.add_collection(lc)

            if mind<mindtol:
                ax.scatter(output[trial_i,-1,0], output[trial_i,-1,1], marker='.', s=100, color=cmap2(norm2(target_angle)), zorder=110)
    if np.any(fxd_points):
        proj_fxd_points = np.dot(fxd_points, wo)
        ax.scatter(proj_fxd_points[:,0], proj_fxd_points[:,1],  marker='s',
                   c=stab_colors[h_stabilities], alpha=.5, zorder=101)    
    if np.any(ops_fxd_points):
        proj_ops_fxd_points = np.dot(ops_fxd_points, wo)
        ax.scatter(proj_ops_fxd_points[:,0], proj_ops_fxd_points[:,1],  marker='x', color='k', alpha=.3, zorder=101)    

    if plot_epoch:
        plt.text(-1.4, 1.4, s=plot_epoch, fontsize=10)

    ax.set_axis_off()
    ax.set_xlim(xylims)
    ax.set_ylim(xylims)
        
def plot_trajs_model(main_exp_name, model_name, exp_i, T=128, which='post',  hidden_i=0,
                     input_length=10, timepart='all',  num_of_inputs=51, plot_from_to=(0,None),
                     pca_from_to=(0,None), input_range=(-3,3), axes=None, limcyctol=1e-2,
                     slowpointtol=1e-8, slowpointmethod='L-BFGS-B'):
    pca_before_t, pca_after_t = pca_from_to
    after_t, before_t = plot_from_to
    norm = mplcolors.Normalize(vmin=-.5,vmax=.5)
    norm = norm(np.linspace(-.5, .5, num=num_of_inputs, endpoint=True))
    cmap = cmx.get_cmap("coolwarm")
    norm2 = mpl.colors.Normalize(-np.pi, np.pi)
    cmap2 = plt.get_cmap('hsv')
    params_folder = parent_dir+'/experiments/' + main_exp_name +'/'+ model_name
    wi, wrec, wo, brec, h0, training_kwargs = get_params_exp(params_folder, exp_i=exp_i)
    
    trajectories, traj_pca, start, target, output, input_proj, pca, explained_variance = get_hidden_trajs(wi, wrec, wo, brec, h0, training_kwargs,
                                                                       T=T, which=which,
                                                                       input_length=input_length, timepart=timepart,
                                                                       num_of_inputs=num_of_inputs, pca_from_to=pca_from_to,
                                                                       input_range=input_range)

    if not axes:
        # fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=False, sharey=False)
        # fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=False, sharey=False)
        fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharex=False, sharey=False)
        # fig, axes = plt.subplots(1, 6, figsize=(18, 3), sharex=False, sharey=False)

    for trial_i in range(trajectories.shape[0]):
        axes[0].plot(traj_pca[trial_i,:,0], traj_pca[trial_i,:,1], color=cmap(norm[trial_i]))

        target_angle = np.arctan2(output[trial_i,-1,1], output[trial_i,-1,0])
        axes[2].plot(output[trial_i,0,0], output[trial_i,0,1], '.', c='k', zorder=100) #starting point
        axes[2].plot(output[trial_i,:input_length,0], output[trial_i,:input_length,1], '-', color=cmap(norm[trial_i]), alpha=.25) #all trajectories
        axes[2].plot(output[trial_i,input_length:,0], output[trial_i,input_length:,1], '--', color=cmap(norm[trial_i]), alpha=.25) #all trajectories
        idx, mind = identify_limit_cycle(trajectories[trial_i,:,:], tol=limcyctol) #find recurrence
        if idx:
            axes[1].plot(traj_pca[trial_i,idx:,0], traj_pca[trial_i,idx:,1], '-', linewidth=3, color=cmap2(norm2(target_angle)), zorder=100)
            axes[2].plot(output[trial_i,idx:,0], output[trial_i,idx:,1], '-', linewidth=3, color=cmap2(norm2(target_angle)), zorder=100)
            axes[3].plot(input_proj[trial_i,after_t:before_t,0], '-', color=cmap2(norm2(target_angle)))

            if mind<1e-4:
                axes[1].scatter(traj_pca[trial_i,-1,0], traj_pca[trial_i,-1,1],  marker='.', s=100, color=cmap2(norm2(target_angle)), zorder=110)
                axes[2].scatter(output[trial_i,-1,0], output[trial_i,-1,1], marker='.', s=100, color=cmap2(norm2(target_angle)), zorder=110)

    axes[1].set_axis_off()
    axes[1].scatter(start[0], start[1], marker='.', s=100, color='k', zorder=100)
        
    traj = trajectories[:,:input_length:1500,:].reshape((-1,training_kwargs['N_rec']))
    print('Finding slow points in hidden space')
    fxd_points, speeds = find_slow_points(wrec, brec, dt=training_kwargs['dt_rnn'], trajectory=traj, tol=slowpointtol, method=slowpointmethod)
    print('Finding slow points in output space')
    ops_fxd_points, speeds = find_slow_points(wrec, brec, wo=wo, dt=training_kwargs['dt_rnn'], trajectory=traj,
                                              outputspace=True, tol=slowpointtol, method=slowpointmethod)

    # print(fxd_points.shape)
    if fxd_points.shape[0]>0:
        pca_fxd_points = pca.transform(fxd_points)
        pca_ops_fxd_points = pca.transform(ops_fxd_points)
        axes[1].scatter(pca_fxd_points[:,0], pca_fxd_points[:,1], marker='s', color='k', alpha=.5, zorder=101)    
        axes[0].scatter(pca_fxd_points[:,0], pca_fxd_points[:,1],  marker='s', color='k', alpha=.5, zorder=101)    
        axes[1].scatter(pca_ops_fxd_points[:,0], pca_ops_fxd_points[:,1],  marker='x', color='k', alpha=.5, zorder=101)    
        axes[0].scatter(pca_ops_fxd_points[:,0], pca_ops_fxd_points[:,1],  marker='x', color='k', alpha=.5, zorder=101)    
        proj_fxd_points = np.dot(fxd_points, wo)
        proj_ops_fxd_points = np.dot(ops_fxd_points, wo)
        axes[2].scatter(proj_fxd_points[:,0], proj_fxd_points[:,1],  marker='s', color='k', alpha=.5, zorder=101)    
        axes[2].scatter(proj_ops_fxd_points[:,0], proj_ops_fxd_points[:,1],  marker='x', color='k', alpha=.5, zorder=101)    

    
    x = np.linspace(-np.pi, np.pi, 1000)
    # axes[1].plot(np.cos(x), np.sin(x), 'k', alpha=.5, linewidth=5, zorder=-1)
    axes[2].scatter(np.cos(x), np.sin(x), color=cmap2(norm2(x)), alpha=.5, s=2, zorder=-1) #ring

    #Speed
    x_lim = np.max(np.abs(output.numpy()))
    num_x_points = 21
    # output_logspeeds, all_logspeeds = average_logspeed(wrec, wo, brec, trajectories[:,input_length:,:], x_min=-x_lim, x_max=x_lim, num_x_points=num_x_points)
    # im=axes[4].imshow(np.rot90(output_logspeeds), cmap='inferno')
    # cbar = fig.colorbar(im, ax=axes[3])
    # cbar.set_label("log(speed)")
    
    #Vector field
    plot_average_vf(trajectories[:1,:input_length,:], wi, wrec, brec, wo, input_length=input_length, 
                    num_of_inputs=num_of_inputs, input_range=(input_range[0],input_range[0]), x_lim=x_lim,
                    num_x_points=num_x_points, color='red', ax=axes[4], change_from_outtraj=False)
    
    plot_average_vf(trajectories[-1:,:input_length,:], wi, wrec, brec, wo, input_length=input_length, 
                    num_of_inputs=num_of_inputs, input_range=(input_range[1],input_range[1]), x_lim=x_lim,
                    num_x_points=num_x_points, color='blue', ax=axes[4], change_from_outtraj=False)
    axes[2].set_xlim([-x_lim, x_lim])
    axes[2].set_ylim([-x_lim, x_lim])

    axes[0].set_axis_off()
    axes[1].set_axis_off()
    axes[2].set_axis_off()
    axes[3].set_xticks([])
    axes[4].set_axis_off()

    plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/mss_output_{which}_{exp_i}.pdf', bbox_inches="tight", pad_inches=0.0)
    plt.show()
    
    

def plot_average_vf(trajectories, wi, wrec, brec, wo, input_length=10,
                     num_of_inputs=51, input_range=(-3,3), change_from_outtraj=False,
                     ax=None, x_lim=1.2, num_x_points=21, color='k', scale=1.):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharex=False, sharey=False)

    x_values = np.linspace(-x_lim, x_lim, num_x_points)
    y_values = np.linspace(-x_lim, x_lim, num_x_points)
    
    # Generate all grid points
    grid_points = np.array([(np.round(x,5), np.round(y,5)) for x in x_values for y in y_values])+(2*x_lim)/num_x_points
    average_vf = np.zeros((num_x_points, num_x_points, 2))
    if change_from_outtraj:
        for trajectory in trajectories:
            outtrajectories = np.dot(trajectory, wo)    
            average_vf += average_output_changevectors(wi, wrec, brec, wo, 0, outtrajectories, x_min=-x_lim, x_max=x_lim, num_x_points=num_x_points)

    elif input_length==0:
        for trajectory in trajectories:
            average_vf += average_input_vectorfield(wi, wrec, brec, wo, 0, trajectory, x_min=-x_lim, x_max=x_lim, num_x_points=num_x_points)
    else:
        for I, trajectory in zip(np.linspace(input_range[0], input_range[1], num=num_of_inputs, endpoint=True), trajectories[:,:input_length,:]):
            # print("I", I)
            average_vf += average_input_vectorfield(wi, wrec, brec, wo, I, trajectory, x_min=-x_lim, x_max=x_lim, num_x_points=num_x_points)
    
    average_vf = np.round(average_vf, 2)
    average_vf /= num_of_inputs

    # print(average_vf)
    theta = np.radians(90)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    average_vf = np.dot(average_vf, R)
    ax.quiver(grid_points[:,0], grid_points[:,1],
                   average_vf.reshape((num_x_points**2,2))[:,0], average_vf.reshape((num_x_points**2,2))[:,1], color=color,
                   angles='xy', scale_units='xy', scale=scale)
    ax.set_axis_off()



###plot network (hidden) dynamics (trajectories)
def get_hidden_trajs(net, T=128, input_length=10, input_type='constant', 
                     num_of_inputs=51, input_range=(-3,3), 
                     pca_from_to=(0,None), n_components=10, 
                     dt_task=1, angle_init=False, task=None,
                     input=None, target=None):
    
    pca_after_t, pca_before_t = pca_from_to
    min_input, max_input = input_range

    input_size, hidden_size, output_size = net.dims
    n_components = min(n_components, hidden_size)

    #Generate input
    if input_type == 'constant':  #generate constant inputs (left/right with a range determined by input_range)
        input = np.zeros((num_of_inputs, T, input_size))
        stim = np.linspace(min_input, max_input, num=num_of_inputs, endpoint=True)
        input[:,:input_length,0] = np.repeat(stim,input_length).reshape((num_of_inputs,input_length))
        input = torch.from_numpy(input).float() 
        
        #Integrate input to get target output
        outputs_1d = np.cumsum(input, axis=1)*dt_task
        
        #random init
        outputs_1d = np.cumsum(input, axis=1)*dt_task
        if angle_init == 'random':
            random_angles = np.random.uniform(-np.pi, np.pi, size=num_of_inputs).astype('f')
            outputs_1d += random_angles[:, np.newaxis, np.newaxis]
            
        if angle_init == 'uniform':
            angles = np.arange(-np.pi, np.pi, 2*np.pi/num_of_inputs)
            outputs_1d += angles[:, np.newaxis, np.newaxis]

        target = np.stack((np.cos(outputs_1d), np.sin(outputs_1d)), axis=-1).reshape((num_of_inputs, T, output_size))

    elif input_type == 'training': 
        input_length = input.shape[1]
        num_of_inputs = input.shape[0]
        all_input = np.zeros((num_of_inputs, T, input_size))
        all_input[:,:input_length,0] = input.squeeze()
        input = torch.from_numpy(all_input).float() 

    elif input_type == 'gp':
        _input, target, mask = task(num_of_inputs)
        input = np.zeros((num_of_inputs, T, input_size))
        input[:,:input_length,:] = _input
        input = torch.from_numpy(input).float() 
    else:
        print("Input type not known")
    
    h = net.h0  
    if net.map_output_to_hidden:
        with torch.no_grad():
            #Get initial hidden activations
            h =  np.dot(target[:,0,:], net.output_to_hidden)
            _target = torch.from_numpy(target).float() 
            output, trajectories = net(input, return_dynamics=True, h_init=h, target=_target)
            
    else:
        h = h.detach().numpy()
        output, trajectories = net(input, return_dynamics=True)

    trajectories = trajectories.detach().numpy()
    output = output.detach().numpy()

    input_proj = np.dot(trajectories, net.wi.detach().numpy().T)

    pca = PCA(n_components=n_components)
    pca.fit(trajectories.reshape((-1,hidden_size)))
    explained_variance = pca.explained_variance_ratio_.cumsum()
    trajectories_tofit = trajectories[:,pca_after_t:pca_before_t,:].reshape((-1,hidden_size))
    pca.fit(trajectories_tofit)
    traj_pca = pca.transform(trajectories.reshape((-1,hidden_size))).reshape((num_of_inputs,-1,n_components))
    h0_pca = pca.transform(net.h0.detach().numpy().reshape((1,-1))).T

    return trajectories, traj_pca, h0_pca, input, target, output, input_proj, pca, explained_variance


#plot output dynamics
def plot_output_vs_target(target, output, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    norm2 = mpl.colors.Normalize(-np.pi, np.pi)
    cmap2 = plt.get_cmap('hsv')
    x = np.linspace(-np.pi, np.pi, 1000)
    ax.plot(np.cos(x), np.sin(x), 'k', alpha=.5, linewidth=5, zorder=-1)
    # ax.scatter(np.cos(x), np.sin(x), color=cmap2(norm2(x)), alpha=.5, s=2, zorder=-1)
    ax.scatter(output[0,0], output[0,1], c='k', zorder=10)
    for t in range(output.shape[0]):
                ax.plot([target[t,0], output[t,0]],
                                 [target[t,1], output[t,1]], '-', 
                                 color=cmap2(norm2(np.arctan2(target[t,1], target[t,0]))))
                if t>0:
                    ax.plot([output[t,0], output[t-1,0]],
                                     [output[t,1], output[t-1,1]], '-', 
                                     color=cmap2(norm2(np.arctan2(target[t,1], target[t,0]))))
    ax.set_axis_off()
    plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/output_vs_target.pdf', bbox_inches="tight")




####move to analysis
def average_output_changevectors(wi, wrec, brec, wo, I, trajectories, x_min=-2, x_max=2, num_x_points=11):
    # Create a regular grid
    x_values = np.linspace(x_min, x_max, num_x_points)
    y_values = np.linspace(x_min, x_max, num_x_points)
    
    # Generate all grid points
    # grid_points = [(np.round(x,5), np.round(y,5)) for x in x_values for y in y_values]
    average_vector_field = np.zeros((num_x_points, num_x_points, 2))
    bin_counts = np.zeros((num_x_points, num_x_points))
     
    for i,x in enumerate(trajectories[:-1,:]):
        cell_containing_point = find_grid_cell(x, x_values, y_values, num_x_points=num_x_points)
        full_vf_at_x = x-trajectories[i+1,:]
        average_vector_field[cell_containing_point] += full_vf_at_x.squeeze()
        bin_counts[cell_containing_point] += 1
    
    avf = average_vector_field/bin_counts.reshape((num_x_points,num_x_points,1))
    avf = np.where(avf==np.nan, 0, avf)
    return avf



def average_input_vectorfield(wi, wrec, brec, wo, I, trajectory, x_min=-2, x_max=2, num_x_points=11):
    """
    Calculates average over x for  g(x,I) = tanh(Wx+b+W_inI) - tanh(Wx+b)
    for a fixed I

    """
    # Create a regular grid
    x_values = np.linspace(x_min, x_max, num_x_points)
    y_values = np.linspace(x_min, x_max, num_x_points)
    
    # Generate all grid points
    # grid_points = [(np.round(x,5), np.round(y,5)) for x in x_values for y in y_values]
    average_vector_field = np.zeros((num_x_points, num_x_points, 2))
    bin_counts = np.ones((num_x_points, num_x_points))

    for t, x in enumerate(trajectory):
        target_point = np.dot(wo.T, x)

        cell_containing_point = find_grid_cell(target_point, x_values, y_values, num_x_points=num_x_points)

        if I==0:
            full_vf_at_x = np.tanh(np.dot(wrec, x)+brec).squeeze()
        else:
            full_vf_at_x = input_vectorfield(x, wi, wrec, brec, I).squeeze()
        # print(t, target_point, cell_containing_point, np.dot(wo.T, full_vf_at_x).squeeze())

        average_vector_field[cell_containing_point] += np.dot(wo.T, full_vf_at_x).squeeze()
        bin_counts[cell_containing_point] += 1

    avf = average_vector_field/bin_counts.reshape((num_x_points,num_x_points,1))
    # avf = np.where(np.isnan(avf), 0, avf)
    return avf



def input_vectorfield(x, wi, wrec, brec, I):
    """
    Calculates g(x, I) in the split RNN equation \dot x = f(x)+g(x,I)
    for \dot x = tanh(Wx+b+W_inI)
    i.e. g(x,I) = tanh(Wx+b+W_inI) - tanh(Wx+b)

    """
    dotx = np.tanh(np.dot(wrec, x)+brec+np.dot(wi,I))
    fx = np.tanh(np.dot(wrec, x)+brec)
    return dotx - fx



def find_grid_cell(target_point, x_values, y_values, num_x_points=11):
    
    # Iterate through grid cells and find the cell containing the target point
    for i in range(num_x_points - 1):
        for j in range(num_x_points - 1):
            x1, x2 = x_values[i], x_values[i + 1]
            y1, y2 = y_values[j], y_values[j + 1]
            
            if x1 <= target_point[0] <= x2 and y1 <= target_point[1] <= y2:
                cell_containing_point = (i, j)
                break
    return cell_containing_point
    


def average_logspeed(wrec, wo, brec, trajectories, x_min=-2, x_max=2, num_x_points=11):
    """
    Calculate the average speed over the trajectories per grid element in output space

    """
    
    #discretize output space
    # Create a regular grid
    x_values = np.linspace(x_min, x_max, num_x_points)
    y_values = np.linspace(x_min, x_max, num_x_points)
    
    # Generate all grid points
    grid_points = [(np.round(x,5), np.round(y,5)) for x in x_values for y in y_values]
    output_logspeeds = np.zeros((num_x_points, num_x_points))
    bin_counts = np.zeros((num_x_points, num_x_points))
    
    #calculate speed along trajectories
    all_logspeeds = np.zeros((trajectories.shape[0], trajectories.shape[1]-1))
    for trial_i in range(trajectories.shape[0]):
        # speeds = np.linalg.norm(trajectories[trial_i, :-1, :]-trajectories[trial_i, 1:, :])
        # all_logspeeds[trial_i,:] = np.log(speeds)

        for t in range(trajectories.shape[1]-1):
            x = trajectories[trial_i, t, :]
            # speed = np.linalg.norm(np.dot(wrec,np.tanh(x))+brec - x)
            
            speed = np.linalg.norm(trajectories[trial_i, t+1, :]-trajectories[trial_i, t, :])
            
            all_logspeeds[trial_i,t] = np.log(speed)
            
            target_point = np.dot(wo.T, x)
            cell_containing_point = find_grid_cell(target_point, x_values, y_values, num_x_points=num_x_points)
            output_logspeeds[cell_containing_point] += np.log(speed)
            bin_counts[cell_containing_point] += 1
    return output_logspeeds/bin_counts, all_logspeeds




###speed
def plot_output_speeds(recurrences):
    #TODO: use unrounded recs
    recs = np.unique(np.round(recurrences, 1), axis=0).squeeze()
    
    closest_two = []
    nrecs = recs.shape[0]
    for i in range(nrecs):
        exclude = []
        exclude.append(i)
        incl = np.delete(np.arange(0,nrecs,1),exclude)    
        idx1 = sklearn.metrics.pairwise_distances_argmin_min(recs[i].reshape(1,-1), np.delete(recs, exclude, axis=0))[0][0]
        idx1 = incl[idx1]
        exclude.append(idx1)
        incl = np.delete(np.arange(0,nrecs,1),exclude)
        idx2 = sklearn.metrics.pairwise_distances_argmin_min(recs[i].reshape(1,-1), np.delete(recs, exclude, axis=0))[0][0]
        idx2 = incl[idx2]
        closest_two.append([idx1, idx2])
    
    all_trajs = []
    all_outputs = []
    inter_steps = 20
    input = np.zeros((1, 1024, 2)); input = torch.from_numpy(input).float()

    for i in range(nrecs):
        for j in range(0,inter_steps+1):
            for k in range(2):
                h =  (j*recs[i,:]+(inter_steps-j)*recs[closest_two[i][k],:])/inter_steps
                output, traj1 = net(input, return_dynamics=True, h_init=h)
                traj1 = traj1.detach().numpy()
                all_trajs.append(traj1)
                all_outputs.append(output.detach().numpy())
    all_trajs = np.array(all_trajs).squeeze()
    traj_pca = pca.transform(all_trajs.reshape((-1,training_kwargs['N_rec']))).reshape((all_trajs.shape[0],-1,10))
    plot_inputdriven_trajectory_3d(traj_pca[:,:,:], T, plot_traj=True,
                                        recurrences=None, recurrences_pca=None, wo=wo,
                                        elev=45, azim=135, lims=[xlim, ylim, zlim], plot_epoch=which,
                                        cmap=cmap)
    
    lowspeed_idx = []
    all_speeds = []
    for trial_i in range(all_trajs.shape[0]):
        speeds = []
        for t in range(all_trajs.shape[1]-1):
            
            speed = np.linalg.norm(all_trajs[trial_i, t+1, :]-all_trajs[trial_i, t, :])
            speeds.append(speed)
        all_speeds.append(speeds)
        try:
            lowspeed_idx.append(np.min(np.where(np.array(speeds)<1e-3)))
        except:
            lowspeed_idx.append(10)
            
    all_speeds = np.array(all_speeds)
    
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    for i in range(all_outputs.shape[0]):
        sc = ax.scatter(all_outputs[i,lowspeed_idx[i]:,0].flatten(), all_outputs[i,lowspeed_idx[i]:,1].flatten(), c=np.log(all_speeds[i,lowspeed_idx[i]-1:]))
    clb=fig.colorbar(sc, cax=cax, orientation='vertical', )
    clb.ax.set_title('log(speed)',fontsize=8)
    ax.set_axis_off()
    plt.savefig(output_folder_path+f'/speed_{exp_i}_{which:04d}.png', bbox_inches="tight", pad_inches=0.0)
    
    

#plot error
def plot_error_over_time(params_folder, exp_i, T=None, sparsity=0.2, batch_size=2**8, ax=None, label=None):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3));
    params_folder = parent_dir+'/experiments/' + main_exp_name +'/'+ model_name
    wi, wrec, wo, brec, h0, training_kwargs = get_params_exp(params_folder, exp_i=exp_i)

    dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
    net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'],
              nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
              wi_init=wi, wrec_init=wrec, wo_init=wo, brec_init=brec, h0_init=h0, ML_RNN=training_kwargs['ml_rnn'])
    if not T:
        T = training_kwargs['T']
    # task = angularintegration_task(T=T, dt=training_kwargs['dt_task'], sparsity=sparsity)
    # input, target, mask, output, loss = run_net(net, task, batch_size=batch_size, return_dynamics=False)
    trajectories, traj_pca, start, target, output, input_proj, pca, explained_variance = get_hidden_trajs(wi, wrec, wo, brec, h0, training_kwargs,
                                                                                      T=T, input_length=T, which=which,
                                                                                      pca_from_to=pca_from_to,
                                                                                      num_of_inputs=num_of_inputs, input_range=(-.5,.5))
    #plot error over time
    errors = np.linalg.norm(output-target, axis=(2))
    ax.plot(np.mean(errors,axis=0), label=label)
    # error = np.mean(errors)

def plot_error_over_time_list(main_exp_name, model_name, T=None, batch_size=2**8, 
                              act_lambdas = [0.01, 0.001, 1e-05, 1e-07, 1e-09, 0]):
    colors = [plt.cm.jet(i) for i in np.linspace(.5, 1, len(act_lambdas))]
    fig, ax = plt.subplots(1, 1, figsize=(3, 3));
    ax.set_prop_cycle('color', colors)
    for act_lambda in tqdm(act_lambdas):
        np.random.seed(0)
        main_exp_name_an = main_exp_name + f'/{act_lambda}'
        exp_list = glob.glob(parent_dir+"/experiments/" + main_exp_name_an +'/'+ model_name + "/result*")
        exp = exp_list[0]
        # print(main_exp_name_an)
        if act_lambda==0:
            label = 0
        else:
            label=r'$10^{%02d}$' %np.log10(act_lambda)
        plot_error_over_time(main_exp_name_an, model_name, exp, T=T, batch_size=batch_size, ax=ax, label=label)
    ax.set_xlabel("Time")
    ax.set_ylabel("Error")
    plt.legend(title='Reg. param.', bbox_to_anchor=(1., 1.))
    plt.savefig(parent_dir+f"/experiments/angular_integration/act_norm/error_T{T}.pdf", bbox_inches="tight")


#explained vriance
def plot_explained_variances(main_exp_name, model_name, input_length, batch_size=2**8, 
                              act_lambdas = [0.01, 0.001, 1e-05, 1e-07, 1e-09, 0]):
    colors = [plt.cm.jet(i) for i in np.linspace(.5, 1, len(act_lambdas))]
    fig, ax = plt.subplots(1, 1, figsize=(3, 3));
    ax.set_prop_cycle('color', colors)
    for act_lambda in tqdm(act_lambdas):
        np.random.seed(0)
        main_exp_name_an = main_exp_name + f'/{act_lambda}'
        print(parent_dir+"/experiments/" + main_exp_name_an +'/'+ model_name + "/result*")
        exp_list = glob.glob(parent_dir+"/experiments/" + main_exp_name_an +'/'+ model_name + "/result*")
        exp = exp_list[0]
        if act_lambda==0:
            label = 0
        else:
            label=r'$10^{%02d}$' %np.log10(act_lambda)
        # explained_variance
        params_folder = parent_dir+'/experiments/' + main_exp_name_an +'/'+ model_name
        wi, wrec, wo, brec, h0, training_kwargs = get_params_exp(params_folder, exp_i=0)
        trajectories, traj_pca, start, target, output, input_proj, pca, explained_variance = get_hidden_trajs(wi, wrec, wo, brec, h0, training_kwargs,
                                                                                          T=2*input_length, input_length=input_length, which=which,
                                                                                          pca_from_to=pca_from_to,
                                                                                          num_of_inputs=num_of_inputs, input_range=(-.5,.5))
        ax.plot(explained_variance, label=label)
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Explained variance")
        plt.legend(title='Reg. param.', bbox_to_anchor=(1., 1.))
        plt.savefig(parent_dir+"/experiments/angular_integration/act_norm/exp_vars.pdf", bbox_inches="tight")
        


###learning trajectory
def plot_learning_trajectory(exp_name, exp_i, T=256, num_of_inputs=11, xylims=[-1.5,1.5],
                             input_length=128, input_type='constant', input_range = (-.5, .5),     
                             angle_init=False, task=None,
                             slowpointmethod= 'L-BFGS-B', set_cmap=None):

    params_folder = parent_dir+'/experiments/' + exp_name +'/'
    losses, gradient_norms, epochs, rec_epochs, weights_train = get_traininginfo_exp(params_folder, exp_i, 'post')
    
    net, training_kwargs = load_net(exp_name, exp_i, 'post')
    input = None
    target= None
        
    if input_type=='constant':
        inputdriven_folder = '/inputdriven3d_asymp'
        output_folder = '/output_asymp'
        cmap = cmx.get_cmap("coolwarm")
    
    elif input_type=='training': 
        inputdriven_folder = '/training_inputdriven3d_asymp'
        output_folder = '/training_output_asymp'
        exp_file = glob.glob(parent_dir+"/experiments/" + exp_name + "/result_*")[0]
        with open(exp_file, 'rb') as handle:
            result = pickle.load(handle)
        
        input = torch.from_numpy(result['all_inputs'][0]).float() 
        training_input_length = input.shape[1]
        target = result['all_targets'][0]
        num_of_inputs = input.shape[0]
        max_epoch = np.argmin(losses)
        cmap = cmx.get_cmap("tab10")
    
    else:
        inputdriven_folder = '/gp_inputdriven3d_asymp'
        output_folder = '/gp_output_asymp'
        cmap = cmx.get_cmap("tab10")
        
    if set_cmap:
        cmap = set_cmap

        
    trajectories, traj_pca, start, input, target, output, input_proj, pca, explained_variance = get_hidden_trajs(net, dt_task=training_kwargs['dt_task'], 
                                                                                        T=T, input_length=input_length, 
                                                                                        num_of_inputs=num_of_inputs,
                                                                                        input_range=input_range,
                                                                                        input_type=input_type,
                                                                                        angle_init=angle_init, 
                                                                                        task=task, 
                                                                                        input=input,
                                                                                        target=target)
    xlim = [np.min(traj_pca[...,0]), np.max(traj_pca[...,0])]
    ylim = [np.min(traj_pca[...,1]), np.max(traj_pca[...,1])]
    zlim = [np.min(traj_pca[...,2]), np.max(traj_pca[...,2])]

    makedirs(parent_dir+'/experiments/'+exp_name+'/' + inputdriven_folder)
    makedirs(parent_dir+'/experiments/'+exp_name+'/' + output_folder)
    
    for which in tqdm(np.concatenate([np.arange(0, 100, 1), np.arange(100, 500, 5), np.arange(500, np.argmin(losses), 25)])):
        print(which, parent_dir+'/experiments/'+exp_name+'/' + inputdriven_folder)
    # for which in tqdm(np.arange(4000, max_epoch, 25)):
        # num_of_inputs = 11
        if input_type=='training': 
            input = torch.from_numpy(result['all_inputs'][which]).float() 
            target = result['all_targets'][which]
            # output = result['all_outputs'][which]
            # trajectories = result['all_trajectories'][which]

        net, training_kwargs = load_net(exp_name, exp_i, which)
        wo = net.wo.detach().numpy()
        
        # if input_type != 'training': 
        trajectories, traj_pca, start, input, target, output, input_proj, _, explained_variance = get_hidden_trajs(net, dt_task=training_kwargs['dt_task'], T=T, input_length=input_length, num_of_inputs=num_of_inputs, input_range=input_range, input_type=input_type, input=input, target=target, task=task)

        traj_pca = pca.transform(trajectories.reshape((-1,training_kwargs['N_rec']))).reshape((num_of_inputs,-1,pca.n_components))
        recurrences, recurrences_pca = find_periodic_orbits(trajectories, traj_pca, limcyctol=1e-2, mindtol=1e-4)
        
        plot_inputdriven_trajectory_3d(traj_pca, input_length, plot_traj=True,
                                            recurrences=recurrences, recurrences_pca=recurrences_pca, wo=wo,
                                            elev=45, azim=135, lims=[xlim, ylim, zlim], plot_epoch=which,
                                            cmap=cmap)
        plt.savefig(parent_dir+'/experiments/'+exp_name+'/' + inputdriven_folder + f'/{exp_i}_{which:04d}.png', bbox_inches="tight")
        plt.close()
        
        plot_output_trajectory(trajectories, wo, input_length, plot_traj=True,
                                   fxd_points=None, ops_fxd_points=None,
                                   plot_asymp=True, limcyctol=1e-2, mindtol=1e-4, ax=None, xylims=xylims, plot_epoch=which,
                                   cmap=cmap)
        plt.savefig(parent_dir+'/experiments/'+exp_name+'/' + output_folder + f'/{exp_i}_{which:04d}.png', bbox_inches="tight")
        plt.close()
        
def plot_weights_during_learning(exp_name, parent_dir, exp_i):
    print(exp_name)
    makedirs(parent_dir+'/experiments/'+exp_name+'/' + 'weights_folder')
    params_folder = parent_dir+'/experiments/' + exp_name +'/'
    losses, gradient_norms, epochs, rec_epochs, weights_train = get_traininginfo_exp(params_folder, exp_i, 'post')
    
    wrec = weights_train['wrec']
    N_rec = wrec.shape[-1]

    plt.style.use('seaborn-dark-palette')
    # for i in range(0, np.argmin(losses)):
    for i in tqdm(np.concatenate([np.arange(0, 100, 1), np.arange(100, 500, 5), np.arange(500, np.argmin(losses)-1, 25)])):
        
        plt.scatter(range(wrec.shape[1]**2), wrec[i].flatten(), s=1)
        
        plt.xticks([])
        plt.ylim(np.min(wrec), np.max(wrec))
        plt.ylabel(r"$W_{rec}$")
        
        plt.text(int(N_rec**2/2), np.min(wrec)*.9, s=i, fontsize=10)
        
        plt.savefig(parent_dir+'/experiments/'+exp_name+'/' + 'weights_folder' + f'/{i:04d}.png', bbox_inches="tight")
        plt.close()

def tuning_curves(trajectories, target, nbins=100, plot_fig=False):
    
    #discretize angles
    bins = np.arange(-np.pi, np.pi, np.pi/int(nbins/2))
    angles = np.arctan2(target[...,1],target[...,0])
    
    d_angles = np.digitize(angles, bins=bins)
    
    tunings = np.zeros((nbins,trajectories.shape[-1]))
    for i in range(1,nbins+1):
        tunings[i-1] = np.mean(trajectories[np.where(d_angles==i)], axis=0)
        
    if plot_fig:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3));
        ax.plot(bins, tunings)
        ax.set_xlabel("Angle")
        ax.set_ylabel("Average activity") 
        ax.set_xticks([-np.pi,np.pi], [r'$-\pi$', r'$\pi$'])
        fig.show()
        # plt.close()
        
    return bins, tunings

def plot_centered_tuning(trajectories, tunings=None, target=None, nbins=10, window_size=1, exp_name=None):
    """
    Plot centered tuning curves from generated trajectories on angular integration task.

    Parameters
    ----------
    trajectories : array
        DESCRIPTION.
    tunings : TYPE
        DESCRIPTION.
    centering_method : TYPE
        DESCRIPTION.
    nbins: int
    Number of bins for angles 
    average_k: int > 0
    if 1 then use maximal activity for centeringmethod 
    if more than 1 use average over k

    Returns
    -------

    """
    #assert average_k > 0
    folder = parent_dir+'/experiments/'+exp_name+'tuning_folder'
    makedirs(folder)

    if not np.any(tunings):
        bins, tunings = tuning_curves(trajectories, target, nbins=nbins, plot_fig=False)
    bins = np.arange(-np.pi, np.pi, np.pi/int(nbins/2))

    fig, ax = plt.subplots(1, 1, figsize=(5, 3));

    if window_size==1:
        for i in range(trajectories.shape[-1]):
            ax.plot(bins,   tunings[:,i])
        ax.set_xlabel("Angle")
        ax.set_ylabel("Average activity") 
        ax.set_xticks([-np.pi,np.pi], [r'$-\pi$', r'$\pi$'])
        plt.savefig(folder+'/tuning.pdf', bbox_inches="tight")
        # plt.close()
            
    else:
        for i in range(trajectories.shape[-1]):
            # ax.plot(bins, np.roll(tunings[:,i], -np.argmax(tunings[:,i])+int(nbins/2)))
    
            arr = tunings[:,i]
            num_sequences = len(arr) - window_size + 1
    
            # Initialize array to store averages
            averages = np.zeros(num_sequences)
    
            # Calculate averages for each overlapping sequence
            window_shift = int(window_size/2)
            for j in range(num_sequences):
                averages[j] = np.mean(arr[j-window_shift:j+window_shift])
                
            ax.plot(bins, np.roll(tunings[:,i], -np.argmax(averages)-int(window_size/2)+int(nbins/2)))
        ax.set_xlabel("Angle")
        ax.set_ylabel("Average activity")    
        ax.set_xticks([-np.pi,np.pi], [r'$-\pi$', r'$\pi$'])
        plt.savefig(folder+'/tuning_centered.pdf', bbox_inches="tight")
        # plt.close()





# if False:
# # if __name__ == "__main__":

#     model_names=['lstm', 'low','high', 'ortho', 'qpta']
#     mean_colors = ['k', 'r',  'darkorange', 'g', 'b']
#     trial_colors = ['gray', 'lightcoral', 'moccasin', 'lime', 'deepskyblue']
#     labels=['LSTM', r'$g=.5$', r'$g=1.5$', 'Orthogonal', 'QPTA']
    
#     # main_exp_name='angularintegration/lambda_T2'
#     # plot_final_losses(main_exp_name, exp="triallength", mean_colors=mean_colors, trial_colors=trial_colors, labels=labels)
#     # main_exp_name='angularintegration/sizes2'
#     # plot_final_losses(main_exp_name, exp="size", mean_colors=mean_colors, trial_colors=trial_colors, labels=labels)
    
#     # T = 20
#     # from_t_step = 90
#     # plot_allLEs_model(main_exp_name, 'qpta', which='pre', T=10, from_t_step=0, mean_color='b', trial_color='b', label='', ax=None, save=True)
#     T = 128*32  
#     input_length = int(128)*2
#     num_of_inputs = 111
#     angle_init = 'random'
#     input_range = (-.2, .2)   
#     input_type = 'constant'

#     plot_from_to = (T-1*input_length,T)
#     pca_from_to = (0,input_length)
#     slowpointmethod= 'L-BFGS-B' #'Newton-CG' #
    
#     cmap = cmx.get_cmap("tab10")
#     cmap = cmx.get_cmap("coolwarm")

#     model_name = ''
#     # main_exp_name='angular_integration/hidden/25.6'
#     main_exp_name='angular_integration/N500_T128/recttanh/' #
#     main_exp_name='angular_integration/recttanh_N50_T128/' #
#     # main_exp_name='angular_integration/N50_T128_noisy/tanh' #

#     exp_i = 0
    
#     params_folder = parent_dir+"/experiments/" + main_exp_name +'/'+ model_name
#     exp_list = glob.glob(params_folder + "/result*")    
#     params_path = glob.glob(params_folder + '/param*.yml')[0]
#     exp = exp_list[exp_i]
#     training_kwargs = yaml.safe_load(Path(params_path).read_text())
#     with open(exp, 'rb') as handle:
#         result = pickle.load(handle)

#     if True:    
#         np.random.seed(1234)

#         params_folder = parent_dir+'/experiments/' + main_exp_name +'/'+ model_name
#         which = 'post'
#         losses, gradient_norms, epochs, rec_epochs, weights_train = get_traininginfo_exp(params_folder, exp_i, which)
#         print("Epochs trained: ", np.argmin(losses))
        
#         # input = torch.from_numpy(result['all_inputs'][np.argmin(losses)]).float() 
#         # target = result['all_targets'][np.argmin(losses)]
#         # num_of_inputs = input.shape[0]
#         input = None
#         target= None

#         net, training_kwargs = load_net(main_exp_name, exp_i, 'post')
#         net.noise_std = 0 #1e-2
#         # training_kwargs['noise_std'] = 0
#         wi, wrec, wo, brec, h0, oth, training_kwargs = get_params_exp(params_folder)

#         task =  angularintegration_task(T=input_length*training_kwargs['dt_task'], dt=training_kwargs['dt_task'], sparsity=1, length_scale=1,
#                                         last_mses=training_kwargs['last_mses'], angle_init=angle_init, max_input=.5)
        
#         trajectories, traj_pca, start, input, target, output, input_proj, pca, explained_variance = get_hidden_trajs(net, dt_task=training_kwargs['dt_task'], 
#                                                                                             T=T, input_length=input_length, 
#                                                                                             pca_from_to=pca_from_to,
#                                                                                             num_of_inputs=num_of_inputs,
#                                                                                             input_range=input_range,
#                                                                                             input_type=input_type,
#                                                                                             angle_init=angle_init, 
#                                                                                             task=task,
#                                                                                             input=input,
#                                                                                             target=target)
        
#         nbins = 40
#         tunings = tuning_curves(trajectories[:,:input_length,:], target[:,:input_length,:], nbins=nbins, plot_fig=False)

#         plot_centered_tuning(trajectories, tunings=tunings, nbins=nbins, window_size=1)
#         plot_centered_tuning(trajectories, tunings=tunings, nbins=nbins, window_size=5)

#         xlim = [np.min(traj_pca[...,0]), np.max(traj_pca[...,0])]
#         ylim = [np.min(traj_pca[...,1]), np.max(traj_pca[...,1])]
#         zlim = [np.min(traj_pca[...,2]), np.max(traj_pca[...,2])]
        
#     rcParams["figure.dpi"] = 250
#     plt.rcParams["figure.figsize"] = (5,5)
#     xylims=[-1.5,1.5]
#     inputdriven_folder = '/inputdriven3d_asymp'
#     output_folder = '/output_asymp'
#     vf_folder = '/vf'

#     inputdriven_folder_path = parent_dir+'/experiments/'+main_exp_name+'/'+ model_name + inputdriven_folder
#     output_folder_path = parent_dir+'/experiments/'+main_exp_name+'/'+ model_name + output_folder
#     vf_folder_path = parent_dir+'/experiments/'+main_exp_name+'/'+ model_name + vf_folder

#     makedirs(inputdriven_folder_path)
#     makedirs(output_folder_path)
#     makedirs(vf_folder_path)

#     if training_kwargs['nonlinearity'] == 'relu' and training_kwargs['N_rec'] <= 25:
#         makedirs(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +'/inputdriven3d_analytic')
#         makedirs(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +'/output_analytic')
    
#     for which in [np.argmin(losses)]:
        
#     # for which in ['post']:
#     # for which in range(500, np.argmin(losses), 25):
#     # for which in  range(100, 500, 5):
#     # for which in range(0, 100, 1):
        
#         # input = torch.from_numpy(result['all_inputs'][which]).float() 
#         # target = result['all_targets'][which]
#         # num_of_inputs = input.shape[0]
#         # input = None
#         # target= None
        
#         # net, training_kwargs = load_net(main_exp_name, exp_i, which)
#         wo = net.wo.detach().numpy()
        
#         # trajectories, traj_pca, start, input, target, output, input_proj, _, explained_variance = get_hidden_trajs(net, dt_task=training_kwargs['dt_task'], 
#         #                                                                                     T=T, input_length=input_length,
#         #                                                                                     pca_from_to=pca_from_to, 
#         #                                                                                     num_of_inputs=num_of_inputs,
#         #                                                                                     input_range=input_range, 
#         #                                                                                     input_type=input_type,
#         #                                                                                     angle_init=angle_init,
#         #                                                                                     task=task,
#         #                                                                                     input=input,
#         #                                                                                     target=target)
        
#         traj_pca = pca.transform(trajectories.reshape((-1,training_kwargs['N_rec']))).reshape((num_of_inputs,-1,10))
#         recurrences, recurrences_pca = find_periodic_orbits(trajectories, traj_pca, limcyctol=1e-2, mindtol=1e-4)
        
#         # plot_input_driven_trajectory_2d(trajectories, traj_pca, wo, ax=None)
#         # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/inputdriven2d_{which}_{exp_i}.pdf', bbox_inches="tight")
        
#         # plot_input_driven_trajectory_2d(trajectories, traj_pca, wo, plot_asymp=True, ax=None);
#         # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/inputdriven2d_asymp_{which}_{exp_i}.pdf', bbox_inches="tight")
        
#         # id_recurrences, id_recurrences_pca = find_periodic_orbits(trajectories[:,:input_length,:], traj_pca[:,:input_length,:], limcyctol=1e-1, mindtol=1e-4)
        

#         plot_inputdriven_trajectory_3d(traj_pca, input_length, plot_traj=True,
#                                             recurrences=recurrences, recurrences_pca=recurrences_pca, wo=wo,
#                                             elev=45, azim=135, lims=[xlim, ylim, zlim], plot_epoch=which,
#                                             cmap=cmap)
#         plt.savefig(inputdriven_folder_path+f'/{exp_i}_{which:04d}.png', bbox_inches="tight", pad_inches=0.0)
#         plt.close()
        
#         plot_output_trajectory(trajectories, wo, input_length, plot_traj=True,
#                                    fxd_points=None, ops_fxd_points=None,
#                                    plot_asymp=True, limcyctol=1e-2, mindtol=1e-4, ax=None, xylims=xylims, plot_epoch=which,
#                                    cmap=cmap)
#         plt.savefig(output_folder_path+f'/{exp_i}_{which:04d}.png', bbox_inches="tight", pad_inches=0.0)
#         plt.close()

#         # plot_output_trajectory(trajectories, wo, input_length, plot_traj=False,
#         #                            fxd_points=None, ops_fxd_points=None,
#         #                            plot_asymp=True, limcyctol=1e-2, mindtol=1e-4, ax=None)
#         # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/output_asymp_notraj_{which}_{exp_i}.pdf', bbox_inches="tight")
        
#         # diagrams, fig, ax = tda_inputdriven_recurrent(id_recurrences, maxdim=1)
#         # plot_diagrams(diagrams, show=True)
#         # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/inputdriven_tda_{exp_i}.png', bbox_inches="tight")
#         # plt.close()

#         if training_kwargs['nonlinearity'] == 'relu' and training_kwargs['N_rec'] <= 15:
        
#             fixed_point_list, stabilist, unstabledimensions, eigenvalues_list = find_analytic_fixed_points(wrec, brec)
            
#             plot_inputdriven_trajectory_3d(traj_pca, input_length, plot_traj=True,
#                                                 recurrences=recurrences, recurrences_pca=recurrences_pca, wo=wo,
#                                                 fxd_points=fixed_point_list,
#                                                 h_stabilities=unstabledimensions,
#                                                 elev=45, azim=135, lims=[xlim, ylim, zlim], plot_epoch=which)
#             plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/inputdriven3d_analytic/{exp_i}_{which:04d}.png', bbox_inches="tight")
#             plt.close()

            
#             plot_output_trajectory(trajectories[:,:,:], wo, input_length, plot_traj=True,
#                                        fxd_points=fixed_point_list,
#                                        h_stabilities=unstabledimensions,
#                                        plot_asymp=True, limcyctol=1e-2, mindtol=1e-4, ax=None, xylims=xylims, plot_epoch=which)
#             plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/output_analytic/{exp_i}_{which:04d}.png', bbox_inches="tight")
#             plt.close()
        
#         if False:
#             traj = trajectories[:,:input_length:20,:].reshape((-1,training_kwargs['N_rec']));
#             slowpointtol=1e-27
#             fxd_points, speeds = find_slow_points(wrec, brec, dt=training_kwargs['dt_rnn'], trajectory=traj, tol=slowpointtol, method=slowpointmethod)
            
#             ops_fxd_points, ops_speeds = find_slow_points(wrec, brec, wo=wo, dt=training_kwargs['dt_rnn'], trajectory=traj,
#                                                       outputspace=True, tol=slowpointtol, method=slowpointmethod,
#                                                       nonlinearity=training_kwargs['nonlinearity'])

    
#             h_stabilities=get_stabilities(fxd_points, wrec, brec, tau=1/training_kwargs['dt_rnn'])
#             h_stabilities = np.where(np.array(h_stabilities)>2,2,h_stabilities)
#             plot_input_driven_trajectory_2d(trajectories, traj_pca, wo, fxd_points=fxd_points, 
#                                             ops_fxd_points=ops_fxd_points,
#                                             h_stabilities=h_stabilities, plot_asymp=True, ax=None);
#             plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/inputdriven2d_asymp_slow_{which}_{exp_i}.pdf', bbox_inches="tight")
            
#             plot_output_trajectory(trajectories, wo, input_length, plot_traj=True,
#                                        fxd_points=fxd_points, ops_fxd_points=ops_fxd_points,
#                                        h_stabilities=h_stabilities,
#                                        plot_asymp=True, limcyctol=1e-2, mindtol=1e-4, ax=None)
#             plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/output_slow_{which}_{exp_i}.pdf', bbox_inches="tight")
    
#             plot_output_trajectory(trajectories, wo, input_length, plot_traj=False,
#                                        fxd_points=fxd_points, ops_fxd_points=ops_fxd_points,
#                                        h_stabilities=h_stabilities,
#                                        plot_asymp=True, limcyctol=1e-2, mindtol=1e-4, ax=None)
#             plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/output_slow_notraj_{which}_{exp_i}.pdf', bbox_inches="tight")

#         # plot_trajs_model(main_exp_name, model_name, exp_i, T=T, which='post', input_length=input_length,
#         #                           timepart='all', num_of_inputs=num_of_inputs,
#         #                           plot_from_to=plot_from_to, pca_from_to=pca_from_to,
#         #                           input_range=input_range, slowpointtol=1e-15, slowpointmethod=slowpointmethod)
        
#         # #Vector field plotting
#         # params_folder = parent_dir+'/experiments/' + main_exp_name +'/'+ model_name
#         # wi, wrec, wo, brec, h0, training_kwargs = get_params_exp(params_folder)

#             #Both directions
#             x_lim = 1.4
#             num_x_points = 21
#             num_of_inputs = 11
#             input_range = (-.3,.3)
#             trajectories, traj_pca, start, input, target, output, input_proj, pca, explained_variance = get_hidden_trajs(net, dt_task=training_kwargs['dt_task'], 
#                                                                                             T=T, input_length=input_length, 
#                                                                                             pca_from_to=pca_from_to,
#                                                                                             num_of_inputs=num_of_inputs,
#                                                                                             input_range=input_range,
#                                                                                             input_type=input_type,
#                                                                                             angle_init=angle_init, 
#                                                                                             task=task,
#                                                                                             input=input,
#                                                                                             target=target)
#             plot_average_vf(trajectories[:,input_length:,:], wi, wrec, brec, wo, input_length=0, 
#                             num_of_inputs=num_of_inputs, input_range=input_range, x_lim=x_lim,
#                             num_x_points=num_x_points)
#             plt.savefig(vf_folder_path +f'/vf_noinput_{which}_{exp_i}.pdf', bbox_inches="tight")
#             plt.show()
            
#             #Single direction
#             fig, ax = plt.subplots(1, 1, figsize=(3, 3));
#             input_range = (.5, .51)
#             num_of_inputs = 1
#             trajectories, traj_pca, start, input, target, output, input_proj, pca, explained_variance = get_hidden_trajs(net, dt_task=training_kwargs['dt_task'], 
#                                                                                             T=T, input_length=input_length, 
#                                                                                             pca_from_to=pca_from_to,
#                                                                                             num_of_inputs=num_of_inputs,
#                                                                                             input_range=input_range,
#                                                                                             input_type=input_type,
#                                                                                             angle_init=angle_init, 
#                                                                                             task=task,
#                                                                                             input=input,
#                                                                                             target=target)
        
#             plot_average_vf(trajectories, wi, wrec, brec, wo, input_length=T, 
#                             num_of_inputs=num_of_inputs, input_range=input_range, x_lim=x_lim,
#                             num_x_points=num_x_points, color='blue', ax=ax)
            
#             input_range = (-.51, -.5)
#             num_of_inputs = 1
#             trajectories, traj_pca, start, input, target, output, input_proj, pca, explained_variance = get_hidden_trajs(net, dt_task=training_kwargs['dt_task'], 
#                                                                                             T=T, input_length=input_length, 
#                                                                                             pca_from_to=pca_from_to,
#                                                                                             num_of_inputs=num_of_inputs,
#                                                                                             input_range=input_range,
#                                                                                             input_type=input_type,
#                                                                                             angle_init=angle_init, 
#                                                                                             task=task,
#                                                                                             input=input,
#                                                                                             target=target)
        
#             plot_average_vf(trajectories, wi, wrec, brec, wo, input_length=T, 
#                             num_of_inputs=num_of_inputs, input_range=input_range, x_lim=x_lim,
#                             num_x_points=num_x_points, color='red', ax=ax)
#             # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/vf_onedir_{which}_{exp_i}.pdf', bbox_inches="tight")
#             plt.savefig(vf_folder_path+f'/vf_both_{which}_{exp_i}.pdf', bbox_inches="tight")
#             plt.show()
            
            
#             fig, ax = plt.subplots(1, 1, figsize=(3, 3));
#             im=ax.imshow(np.rot90(output_logspeeds), cmap='inferno')
#             ax.set_axis_off()
#             cbar = fig.colorbar(im, ax=ax)
#             cbar.set_label("log(speed)")
#             plt.savefig(output_folder_path+f'/mss_output_{which}_{exp_i}.png', bbox_inches="tight", pad_inches=0.0)
    
#         # fig, ax = plt.subplots(1, 1, figsize=(3, 3));
#         # # for i in range(10):
#         # plot_output_vs_target(target[0,...], output[0,...], ax=ax)
#         # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f"/output_vs_target_{exp_i}.pdf")

#     #plot target vs output
#     # plot_output_vs_target(target[0,...], output[0,...])

    
#     # for trial_i in range(trajectories.shape[0]):
#     #     target_angle = np.arctan2(output[trial_i,-1,1], output[trial_i,-1,0])

#     #     ax.plot(traj_pca[trial_i,after_t:before_t,0], traj_pca[trial_i,after_t:before_t,1], '-', color=cmap2(norm2(target_angle)))




#     # plot_allLEs(main_exp_name,  mean_colors, trial_colors, labels, T=T, from_t_step=from_t_step, first_or_last='first')
#     # plot_allLEs(main_exp_name,  mean_colors, trial_colors, labels, T=T, from_t_step=from_t_step, first_or_last='last')


# # plot_losses_and_trajectories(plot_losses_or_trajectories='losses', main_exp_name='angularintegration', model_name='high_gain')
    
#     # model_names=['lstm', 'low','high', 'ortho', 'qpta']
#     # i, model_names = 3, ['ortho']

#     # plot_all(model_names=model_names,                   main_exp_name='angularintegration/lambda_T2/12.8')
#     # for sparsity in [0.01, 0.1, 1.]:
#     #     plot_bestmodel_traj(model_names=model_names, main_exp_name=main_exp_name, sparsity=sparsity, i=i)
    
#     # experiment_folder = 'lambda_grid_2'
#     # for model_name in model_names:
#     #     load_best_model_from_grid(experiment_folder, model_name)
