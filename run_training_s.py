# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:25:35 2023

@author: 
"""

import os, sys
import glob 
current_dir = os.path.dirname(os.path.realpath('__file__')) 
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) 
import pickle
import time

import yaml
import shutil
from pathlib import Path
from scipy.linalg import qr, block_diag

import numpy as np
import pandas as pd
from itertools import product
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from models import train, mse_loss_masked, get_optimizer, get_loss_function, get_scheduler, RNN
from network_initialization import qpta_rec_weights, bla_rec_weights, perfect_params, perfect_initialization
from qpta_initializers import _qpta_tanh_hh
from tasks import *
from load_network import get_params_exp
from utils import makedirs

def get_task(task_name = 'angular_integration', T=10, dt=.1, t_delay=50, sparsity=1, last_mses=None,
             input_length=0, task_noise_sigma=0., final_loss=False, random_angle_init=True, max_input=None, fixed_step=False, step_size=1,
             cue_output_durations=[5,5,75,5,5], time_until_cue_range=None, input_noise_level=0):
    
    if task_name == 'eyeblink':
        task =  eyeblink_task(input_length=T, t_delay=t_delay)
        
    #1D linear integration
    elif task_name == 'bernoulli_integration':
        task = bernouilli_noisy_integration_task(T=T, dt=dt, input_length=input_length, final_loss=final_loss, input_noise_level=input_noise_level)
        
    elif task_name == 'single_pulse':
        task = singlepulse_integration_task(T=T, dt=dt, input_length=input_length, 
                                            final_loss=final_loss, fixed_step=fixed_step, step_size=step_size)
        
    #1D circular integration
    elif task_name == 'angular_integration':
        task =  angularintegration_task(T=T, dt=dt, sparsity=sparsity,
                                        last_mses=last_mses, random_angle_init=random_angle_init)
    
    #2D
    elif task_name == 'double_angular_integration':
        task =  double_angularintegration_task(T=T, dt=dt, sparsity=sparsity,
                                        last_mses=last_mses, random_angle_init=random_angle_init)
        
    elif task_name == 'sphere_integration':
        task =  sphere_integration_task(T=T, dt=dt, sparsity=sparsity, random_angle_init=random_angle_init)
        
        
    elif task_name == 'poisson_clicks_task':
        task =  poisson_clicks_task(T=T, dt=dt, cue_output_durations=cue_output_durations) #[10,5,10,5,10]

    elif task_name == 'contbernouilli_noisy_integration_task':
        task = contbernouilli_noisy_integration_task(T=T,
                                                  input_length=input_length,
                                                  sigma=task_noise_sigma,
                                                  final_loss=final_loss)
        
    elif task_name == 'center_out': 
        task = center_out_reaching_task(T, dt, 
                                     cue_output_durations=cue_output_durations,
                                     time_until_cue_range=time_until_cue_range)
        
    elif task_name == 'addition':
            task = addition_task(T)
            
    elif task_name == 'multiplication':
        task = multiplication_task(T)
        
    elif task_name == 'integration_2d':
        task = integration_2d_task(T, dt, threshold=1)
        
    elif task_name == 'integration_2d_gridinit':
        task = integration_2d_task(T, dt, threshold=1, random_angle_init='equally_spaced')
        
    else:
        raise Exception("Task not known.")
        
    return task

def run_experiment(parameter_file_name, main_exp_name='', sub_exp_name='', model_name='', trials=10, training_kwargs={}):
    experiment_folder = parent_dir + '/experiments/' + main_exp_name +'/'+ sub_exp_name +'/'+ model_name
    print(parent_dir, main_exp_name, sub_exp_name)
    print("EXPFOLDER", experiment_folder)
    
    makedirs(experiment_folder) 
    
    for trial in tqdm.tqdm(range(trials)):
        run_single_training(parameter_file_name, exp_name= main_exp_name +'/'+ sub_exp_name +'/'+ model_name, trial=trial, training_kwargs=training_kwargs)

def run_single_training(parameter_file_name, exp_name='', trial=None, save=True, params_folder='', training_kwargs={}):
    
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    if training_kwargs['fix_seed']:
        seed = np.random.randint(time.time())
        np.random.seed(seed)
        torch.manual_seed(seed)
        training_kwargs['seed'] = seed
    
    parameter_path = parent_dir + '/experiments/' + parameter_file_name
    if not training_kwargs['record_step']:
        training_kwargs['record_step'] = training_kwargs['n_epochs'] 
    
    experiment_folder = parent_dir + '/experiments/' + exp_name 
    
    if training_kwargs=={}:
        training_kwargs = yaml.safe_load(Path(parameter_path).read_text())
        if save:
            shutil.copy(parameter_path, experiment_folder)
    else:
        if save:
            with open(experiment_folder+'/parameters.yml', 'w') as outfile:
                yaml.dump(training_kwargs, outfile, default_flow_style=False)
        

    if training_kwargs['task']==None:
        task = None
        data = np.load(parent_dir+"/experiments/datasets/"+training_kwargs['dataset_filename'])
    else:
        data = None
        # task = get_task(task_name=training_kwargs['task'], T=training_kwargs['T'], dt=training_kwargs['dt_task'],
        #                 sparsity=training_kwargs['task_sparsity'], last_mses=training_kwargs['last_mses'])
        
        task = get_task(task_name=training_kwargs['task'], T=training_kwargs['T'], dt=training_kwargs['dt_task'],
                        input_length=training_kwargs['input_length'], sparsity=training_kwargs['task_sparsity'],
                        task_noise_sigma=training_kwargs['task_noise_sigma'], last_mses=training_kwargs['last_mses'],
                        random_angle_init=training_kwargs['random_angle_init'], max_input=training_kwargs['max_input'], 
                        fixed_step=training_kwargs['fixed_step'], step_size=training_kwargs['step_size'],
                        time_until_cue_range=training_kwargs['time_until_cue_range'], input_noise_level=training_kwargs['input_noise_level'])

    dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
    
    if training_kwargs['network_type'] == 'lstm_noforget':
        net = LSTM_noforget((training_kwargs['N_in'],training_kwargs['N_rec'],training_kwargs['N_out']),
                            readout_nonlinearity=training_kwargs['readout_nonlinearity'], dropout=training_kwargs['drouput'])
        
        result = train_lstm(net, task=task, n_epochs=training_kwargs['n_epochs'],
              batch_size=training_kwargs['batch_size'], learning_rate=training_kwargs['learning_rate'],
              clip_gradient=training_kwargs['clip_gradient'], cuda=training_kwargs['cuda'], init_states=None,
              loss_function=training_kwargs['loss_function'], final_loss=training_kwargs['final_loss'], last_mses=training_kwargs['last_mses'], 
              optimizer=training_kwargs['optimizer'], momentum=training_kwargs['adam_momentum'], weight_decay=training_kwargs['weight_decay'],
              adam_betas=(training_kwargs['adam_beta1'],training_kwargs['adam_beta2']), adam_eps=1e-8,  #optimizers 
              scheduler=training_kwargs['scheduler'], scheduler_step_size=training_kwargs['scheduler_step_size'], scheduler_gamma=training_kwargs['scheduler_gamma'], 
              verbose=training_kwargs['verbose'], record_step=training_kwargs['record_step'])
        
    elif training_kwargs['network_type'] == 'lstm':
        net = LSTM((training_kwargs['N_in'],training_kwargs['N_rec'],training_kwargs['N_out']), readout_nonlinearity=training_kwargs['readout_nonlinearity'])
        
        result = train_lstm(net, task=task, n_epochs=training_kwargs['n_epochs'],
              batch_size=training_kwargs['batch_size'], learning_rate=training_kwargs['learning_rate'],
              clip_gradient=training_kwargs['clip_gradient'], cuda=training_kwargs['cuda'], init_states=None,
              loss_function=training_kwargs['loss_function'], final_loss=training_kwargs['final_loss'], last_mses=training_kwargs['last_mses'], 
              optimizer=training_kwargs['optimizer'], momentum=training_kwargs['adam_momentum'], weight_decay=training_kwargs['weight_decay'],
              adam_betas=(training_kwargs['adam_beta1'],training_kwargs['adam_beta2']), adam_eps=1e-8, #optimizers 
              scheduler=training_kwargs['scheduler'], scheduler_step_size=training_kwargs['scheduler_step_size'], scheduler_gamma=training_kwargs['scheduler_gamma'], 
              verbose=training_kwargs['verbose'], record_step=training_kwargs['record_step'])
    else:
        wi_init, wo_init, h0_init, oth_init, bo_init = None, None, training_kwargs['h0_init'], None, None
        
        if training_kwargs['initialization_type'] == 'trained':
            wi_init, wrec_init, wo_init, brec_init, h0_init, oth_init, _ = get_params_exp(training_kwargs['network_folder'], exp_i=training_kwargs['trained_exp_i'])

            # wi_init, wrec_init, wo_init, brec_init, h0_init, _ = get_params_exp(training_kwargs['network_folder'])
            
        elif training_kwargs['initialization_type'] == 'gain':
            wrec_init, brec_init = None, None
            
        elif training_kwargs['initialization_type'] == 'irnn':
            wrec_init =  np.identity(training_kwargs['N_rec'])
            brec_init = np.zeros((training_kwargs['N_rec']))
            
        elif training_kwargs['initialization_type'] == 'perfect_irnn':
            wi_init, wrec_init, wo_init, brec_init, bo_init, h0_init  = perfect_params(1, ouput_bias_value=training_kwargs['b_a'], a=training_kwargs['alpha'])
             
        elif training_kwargs['initialization_type'] == 'perfect_ubla':
            wi_init, wrec_init, wo_init, brec_init, bo_init, h0_init  = perfect_params(2, ouput_bias_value=training_kwargs['b_a'], a=training_kwargs['alpha'])
             
        elif training_kwargs['initialization_type'] == 'perfect_bla':
            wi_init, wrec_init, wo_init, brec_init, bo_init, h0_init  = perfect_params(3, ouput_bias_value=training_kwargs['b_a'], a=training_kwargs['alpha'])
            #net = perfect_initialization(2, ouput_bias_value=training_kwargs['b_a'], a=training_kwargs['alpha'])
            
        elif training_kwargs['initialization_type'] == 'bla':
            wrec_init, brec_init =  bla_rec_weights(training_kwargs['N_in'],
                                                    int(training_kwargs['N_rec']/2),
                                                    training_kwargs['N_out'],
                                                    a=training_kwargs['b_a'])
            
        elif training_kwargs['initialization_type'] == 'ubla':
            ubla_mat = np.array(([[0,1],[1,0]]))
            wrec_init = block_diag(*[ubla_mat]*int(training_kwargs['N_rec']/2))
            brec_init = np.array([0]*training_kwargs['N_rec'])

            wrec_init, brec_init =  bla_rec_weights(training_kwargs['N_in'],
                                                    int(training_kwargs['N_rec']/2),
                                                    training_kwargs['N_out'],
                                                    a=training_kwargs['b_a'])
            
            
        elif training_kwargs['initialization_type'] == 'qpta':
            # N_blas = int(training_kwargs['N_rec']/2)
            # wrec_init, brec_init = qpta_rec_weights(N_in=training_kwargs['N_in'], N_blas=N_blas, N_out=training_kwargs['N_in']);
            brec_init = np.zeros(training_kwargs['N_rec'])
            wrec_init = _qpta_tanh_hh()((training_kwargs['N_rec'],training_kwargs['N_rec']))
            
        elif training_kwargs['initialization_type'] == 'ortho':
            H = np.random.randn(training_kwargs['N_rec'], training_kwargs['N_rec'])
            wrec_init, _ = qr(H)
            brec_init = np.array([0]*training_kwargs['N_rec'])
    
        else:
            raise Exception("Recurrent weight initialization not known.")
            
        
        if training_kwargs['wrec_perturbation']:
            wrec_init = wrec_init.astype(np.float64)
            wrec_init += np.random.normal(0,training_kwargs['wrec_init_std'])
        
        net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'], g_in=training_kwargs['g_in'],
                  nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
                  wi_init=wi_init, wrec_init=wrec_init, wo_init=wo_init, brec_init=brec_init, bo_init=bo_init, h0_init=h0_init, oth_init=oth_init,
                  ML_RNN=training_kwargs['ml_rnn'], save_inputs=training_kwargs['save_inputs'],
                  map_output_to_hidden=training_kwargs['map_output_to_hidden'], input_nonlinearity=training_kwargs['input_nonlinearity'])

        result, res_dict = train(net, task=task, data=data, n_epochs=training_kwargs['n_epochs'],
              batch_size=training_kwargs['batch_size'], learning_rate=training_kwargs['learning_rate'],
              clip_gradient=training_kwargs['clip_gradient'], cuda=training_kwargs['cuda'], 
              h_init=training_kwargs['h0_init'], hidden_initial_variance=training_kwargs['hidden_initial_variance'],
              loss_function=training_kwargs['loss_function'], act_reg_lambda=training_kwargs['act_reg_lambda'],
              optimizer=training_kwargs['optimizer'], momentum=training_kwargs['adam_momentum'], weight_decay=training_kwargs['weight_decay'],
              adam_betas=(training_kwargs['adam_beta1'],training_kwargs['adam_beta2']), adam_eps=1e-8, #optimizers 
              scheduler=training_kwargs['scheduler'], scheduler_step_size=training_kwargs['scheduler_step_size'], scheduler_gamma=training_kwargs['scheduler_gamma'], 
              stop_patience=training_kwargs['stop_patience'], stop_min_delta=training_kwargs['stop_min_delta'],
              verbose=training_kwargs['verbose'], record_step=training_kwargs['record_step'], experiment_folder=experiment_folder)
    
    
    #save result (losses, weights, etc)
    if save:
        result.append(training_kwargs)
        res_dict['training_kwargs'] = training_kwargs
        # with open(experiment_folder + '/results_%s.pickle'%timestamp, 'wb') as handle:
        #     pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(experiment_folder + '/result_dict_%s.pickle'%timestamp, 'wb') as handle:
            pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return result
    
def grid_search(parameter_file_name, param_grid, experiment_folder, parameter_path='', sub_exp_name='', trials=1):
    """Perform a grid search for the optimal hyperparameters for training"""

    experiment_path = parent_dir +  '/experiments/' + experiment_folder
    makedirs(experiment_path)
    makedirs(experiment_path + '/' + sub_exp_name)

    assert trials>=1
    if not parameter_path:
        parameter_path = parent_dir +  '/experiments/' + experiment_folder +'/'+ parameter_file_name
    training_kwargs = yaml.safe_load(Path(parameter_path).read_text())
    
    with open(parent_dir +  '/experiments/' + experiment_folder+'/parameters.yml', 'w') as outfile:
        yaml.dump(training_kwargs, outfile, default_flow_style=False)
    
    #create grid of parameters
    keys, values = zip(*param_grid.items())
    all_param_combs = [dict(zip(keys, p)) for p in product(*values)]
    
    # columns = [i for i in range(training_kwargs['n_epochs'])]
    # columns.extend(['final_loss'])
    # columns.extend(keys)
    # columns.extend(['trial'])
    
    columns = ['losses', 'final_loss', 'trial', 'weights_last']
    columns.extend(keys)
    
    #find params that are varied
    varied_params = [key for key in param_grid.keys() if len(param_grid[key])>1]
    
    L = []
    for param_i, param_comb in tqdm.tqdm(enumerate(all_param_combs)):
        print(param_comb, "   #", param_i+1, "/", len(all_param_combs))

        exp_name = ''
        for i in range(len(varied_params)):
            exp_name += '_' + varied_params[i] + str(param_comb[varied_params[i]]) 
        for key in param_comb:
            training_kwargs[key] = param_comb[key]
        
        for trial in tqdm.tqdm(range(trials)):
            print("Trial", trial)
            result = run_single_training(parameter_file_name,
                                        exp_name='',
                                        trial=trial,
                                        save=False,
                                        training_kwargs=training_kwargs)
            losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result
            dat = [losses]
            dat.append(losses[-1])
            dat.append(trial)
            dat.append(weights_last)
            dat.extend(param_comb.values())
            L.append(dat)
            result.append(training_kwargs)
            with open(experiment_path+"/"+sub_exp_name+"/" + 'results%s.pickle'%exp_name+'_'+str(trial), 'wb') as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    df = pd.DataFrame(L, columns = columns)
    param_keys = list(param_grid.keys())
    all_keys = param_keys
    all_keys.extend(['final_loss', 'trial'])
    df_final = df[all_keys]
    
    save_path = parent_dir+'/experiments/'+experiment_folder+'/grid_search_'+sub_exp_name

        
    df.to_pickle(save_path+'.pickle')
    
    min_final_loss = df.iloc[df['final_loss'].idxmin()]
    df_final = df_final.fillna("None")
    min_final_meanloss = df_final.groupby(param_keys).mean().idxmin()
    return df, df_final, min_final_loss, min_final_meanloss

def grid_search_all_models(experiment_folder, model_names, param_grid):
    for model_i, model_name in enumerate(model_names):
        training_kwargs['network_type'] = network_types[model_i]
        training_kwargs['N_rec'] = nrecs[model_i]
        training_kwargs['loss_function'] = loss_functions[model_i]
        training_kwargs['initialization_type'] = initialization_type_list[model_i]
        training_kwargs['rnn_init_gain'] = g_list[model_i]
        df, df_final, min_final_loss, min_final_meanloss = grid_search(parameter_file_name, param_grid=param_grid,
                                                                        experiment_folder=experiment_folder,
                                                                        sub_exp_name=model_name,
                                                                        parameter_path=parameter_path, trials=2)


def trial_length_experiment(main_exp_name, sub_exp_name):
    
    scheduler_step_sizes = [100, 300, 100, 100, 300]
    gammas = [0.75, 0.5, 0.75, 0.75, .5]
    nrecs = [115, 200, 200, 200, 200]
    trial_lengths = [12.8, 51.2, 102.4, 409.6]
    for T in tqdm.tqdm(trial_lengths):
        training_kwargs['T'] = T
        for model_i, model_name in tqdm.tqdm(enumerate(model_names)):
            training_kwargs['network_type'] = network_types[model_i]
            training_kwargs['initialization_type'] = initialization_type_list[model_i]
            training_kwargs['N_rec'] = nrecs[model_i]
            training_kwargs['loss_function'] = loss_functions[model_i]
            training_kwargs['rnn_init_gain'] = g_list[model_i]
            training_kwargs['scheduler_step_size'] = scheduler_step_sizes[model_i]
            training_kwargs['scheduler_gamma'] = gammas[model_i]
            run_experiment('/parameter_files/'+parameter_file_name, main_exp_name=main_exp_name,
                                                                    sub_exp_name=sub_exp_name+f'/T{T}',
                                                                  model_name=model_name, trials=11, training_kwargs=training_kwargs)

def size_experiment(main_exp_name, sub_exp_name):
    
    network_types = ['lstm_noforget', 'rnn', 'rnn', 'rnn', 'rnn']
    model_names = ['lstm', 'low', 'high', 'ortho', 'qpta']
    initialization_type_list =['', 'gain','gain', 'ortho', 'qpta']
    loss_functions = ['mse', 'mse_loss_masked', 'mse_loss_masked', 'mse_loss_masked', 'mse_loss_masked']
    g_list = [0., .5, 1.5, 0., 0.]
    scheduler_step_sizes = [100, 300, 100, 100, 300]
    gammas = [0.75, 0.5, 0.75, 0.75, .75]
    learning_rates = [1e-3, 1e-3, 1e-3, 1e-3, 1e-4]
    
    nrecs_lists = [[2, 6, 6, 6, 6],
                   [8, 16, 16, 16, 16],
                   [32, 58, 58, 58, 58], 
                   [115, 200, 200, 200, 200],
                   [128, 224, 224, 224, 224]]
    
    training_kwargs['T'] = 25.6
    for nrecs_list in tqdm.tqdm(nrecs_lists):

        for model_i, model_name in tqdm.tqdm(enumerate(model_names)):
            training_kwargs['learning_rate'] = learning_rates[model_i]
            training_kwargs['network_type'] = network_types[model_i]
            training_kwargs['initialization_type'] = initialization_type_list[model_i]
            training_kwargs['N_rec'] = nrecs_list[model_i]
            training_kwargs['loss_function'] = loss_functions[model_i]
            training_kwargs['rnn_init_gain'] = g_list[model_i]
            training_kwargs['scheduler_step_size'] = scheduler_step_sizes[model_i]
            training_kwargs['scheduler_gamma'] = gammas[model_i]
            run_experiment('/parameter_files/'+parameter_file_name, main_exp_name=main_exp_name,
                                                                    sub_exp_name=sub_exp_name+f'/{nrecs_list[0]}',
                                                                  model_name=model_name, trials=11, training_kwargs=training_kwargs)
    
    
    
if __name__ == "__main__":
    print(current_dir)
    parameter_file_name = 'params_ang_sparse.yml'
    
    parameter_path = parent_dir + '/experiments/parameter_files/'+ parameter_file_name
    training_kwargs = yaml.safe_load(Path(parameter_path).read_text())
        
    main_exp_name = 'angular_integration/' 
    training_kwargs['task'] = 'angular_integration'
        
    if training_kwargs['task'] == 'integration_2d':
        T = training_kwargs['T']
    else:
        T = int(training_kwargs['T']/training_kwargs['dt_task'])
    N_rec = training_kwargs['N_rec']
    nonlin = training_kwargs['nonlinearity'].replace('_', '')
    sub_exp_name = f'N{N_rec}_T{T}_noisy/{nonlin}'
    
    run_experiment('/parameter_files/'+parameter_file_name, main_exp_name=main_exp_name,
                                                            sub_exp_name=sub_exp_name,
                                                            model_name="", trials=10,
                                                            training_kwargs=training_kwargs)

