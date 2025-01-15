# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 08:22:57 2025

@author: 
"""
import os
import pickle
import glob
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

from models import RNN

def get_params_exp(params_folder, exp_i=0, which='post'):
    
    exp_list = glob.glob(params_folder + "/res*")
    exp = exp_list[exp_i]
    params_path = glob.glob(params_folder + '/param*.yml')[0]
    training_kwargs = yaml.safe_load(Path(params_path).read_text())
    with open(exp, 'rb') as handle:
        result = pickle.load(handle)
        
        if len(result) == 9:
            losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs, training_kwargs = result
        elif len(result) == 8:
            losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result
        elif len(result) == 7:
            losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result


    if len(weights_last) == 6:
        if which=='post':
            wi, wrec, wo, brec, h0, oth = weights_last
        elif which=='pre':
            wi, wrec, wo, brec, h0, oth = weights_init
            
        else:
            return weights_train["wi"][which], weights_train["wrec"][which], weights_train["wo"][which], weights_train["brec"][which], weights_train["h0"][which], weights_train["oths"][which], training_kwargs, losses
        return wi, wrec, wo, brec, h0, oth, training_kwargs, losses

    else:
        if which=='post':
            wi, wrec, wo, brec, h0 = weights_last
        elif which=='pre':
            wi, wrec, wo, brec, h0 = weights_init
        else:
            return weights_train["wi"][which], weights_train["wrec"][which], weights_train["wo"][which], weights_train["brec"][which], weights_train["h0"][which], None,  training_kwargs, losses

        return wi, wrec, wo, brec, h0, None, training_kwargs, losses

    

def load_net_from_weights(wi, wrec, wo, brec, h0, oth, training_kwargs):
    if oth is None:
        training_kwargs['map_output_to_hidden'] = False

    dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
    net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'],
              nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
              wi_init=wi, wrec_init=wrec, wo_init=wo, brec_init=brec, h0_init=h0, oth_init=oth,
              ML_RNN=training_kwargs['ml_rnn'], 
              map_output_to_hidden=training_kwargs['map_output_to_hidden'], input_nonlinearity=training_kwargs['input_nonlinearity'])
    return net



def load_net_path(path, which='post'):
    # folder = parent_dir+"/experiments/" + main_exp_name
    # exp_list = glob.glob(folder + "/res*")
    # exp = exp_list[exp_i]
    with open(path, 'rb') as handle:
        result = pickle.load(handle)
    if which=='post':
        try:
            wi, wrec, wo, brec, h0, oth = result['weights_last']
        except:
            oth = None
            wi, wrec, wo, brec, h0 = result['weights_last']

    elif which=='pre':
        try:
            wi, wrec, wo, brec, h0, oth = result['weights_init']
        except:
            oth = None
            wi, wrec, wo, brec, h0 = result['weights_init']
    
    
    try:    
        net = load_net_from_weights(wi, wrec, wo, brec, h0, oth, result['training_kwargs'])
    except:
        
        net = load_net_from_weights(wi, wrec, wo, brec, h0, oth, result['training_kwargs'])

    return net, result



def load_net(parent_dir, exp_name, exp_i, which):
    params_folder = parent_dir+'/experiments/' + exp_name +'/'

    try:
        wi, wrec, wo, brec, h0, oth, training_kwargs, losses = get_params_exp(params_folder, exp_i, which)
    except:
        wi, wrec, wo, brec, h0, training_kwargs, losses = get_params_exp(params_folder, exp_i, which)
        oth = None
        
    try:
        training_kwargs['map_output_to_hidden']
    except:
        training_kwargs['map_output_to_hidden'] = False
        
            
    try:
        training_kwargs['input_nonlinearity']
    except:
        training_kwargs['input_nonlinearity'] = None
        
    try:
        h0 = training_kwargs['h_init']
    except:
        0
        
    dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
    net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'],
              g=training_kwargs['rnn_init_gain'],
              nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
              wi_init=wi, wrec_init=wrec, wo_init=wo, brec_init=brec,
              h0_init=h0, oth_init=oth,
              ML_RNN=training_kwargs['ml_rnn'], map_output_to_hidden=training_kwargs['map_output_to_hidden'],
              input_nonlinearity=training_kwargs['input_nonlinearity'])
    
    return net, training_kwargs





def get_weights_from_net(net):
    wi = net.wi.detach().numpy()
    wrec = net.wrec.detach().numpy()
    brec = net.brec.detach().numpy()
    wo = net.wo.detach().numpy()
    try:
        oth = net.output_to_hidden.detach().numpy()
    except:
        oth=None
    return wi, wrec, brec, wo, oth




def load_all(parent_dir, exp_name, exp_i, which='post'):
    folder = parent_dir+"/experiments/" + exp_name
    wi, wrec, wo, brec, h0, oth, training_kwargs, losses = get_params_exp(folder, exp_i, which)
    net, _ = load_net(folder, exp_i, which)
    return net, wi, wrec, wo, brec, h0, oth, training_kwargs, losses





#load (meta)data
def load_losses(path):
    net, result = load_net_path(path)
    losses = result['losses']
    return losses


def get_tr_par(training_kwargs):
    #load training parameters
    T = training_kwargs['T']/training_kwargs['dt_task']
    N = training_kwargs['N_rec']
    if training_kwargs['initialization_type']=='gain':
        I = training_kwargs['rnn_init_gain']
    else:
        I = 'irnn'    
    S = training_kwargs['nonlinearity']
    R = training_kwargs['act_reg_lambda']
    M = training_kwargs['ml_rnn']
    clip_gradient = training_kwargs['clip_gradient']
    return T, N, I, S, R, M, clip_gradient


def load_all_losses_folder(folder, df):
    exp_list = glob.glob(folder + "/res*")
    nexps = len(exp_list)
    all_losses = np.empty((nexps,10000))
    all_losses[:] = np.nan
    # if not np.any(df):
    #     df = pd.DataFrame(columns=['T', 'N', 'I', 'S', 'R', 'M', 'trial'])

    for exp_i in range(nexps):
        path = exp_list[exp_i]
        net, result = load_net_path(path)
        losses = load_losses(path)
        losses[np.argmin(losses):] = np.nan
        training_kwargs = result['training_kwargs']
        T, N, I, S, R, M, clip_gradient = get_tr_par(training_kwargs)
        df = df.append({'T': T, 'N': N, 'I': I, 'S': S, 'R': R, 'M': M, 'clip_gradient':clip_gradient,
                        'trial': exp_i,
                        'losses': losses}, ignore_index=True)

    return df


def load_all_losses(folder):
    
    df = pd.DataFrame(columns=['T', 'N', 'I', 'S', 'R', 'M', 'trial'])

    for dirName, subdirList, fileList in os.walk(folder):
        df = load_all_losses_folder(dirName, df=df)
        
    return df
