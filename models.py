# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 17:44:47 2023

@author: 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from warnings import warn
import time
import pickle
        
def mse_loss_masked(output, target, mask):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: idem
    :param mask: idem -- or torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    loss = (mask * (target - output).pow(2)).sum() / mask.sum()
    return loss


def get_optimizer(model, optimizer, learning_rate, weight_decay=0, momentum=0, adam_betas=(0.9, 0.999), adam_eps=1e-08):
    #initialize optimizer
    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=adam_betas, eps=adam_eps)
    else:
        raise Exception("Optimizer not known.")
    return optimizer

    
def get_loss_function(model, loss_function):
    #initialize loss function
    if loss_function == "mse_loss_masked":
        loss_fn = mse_loss_masked
    elif loss_function == "mse":
        loss_fn = nn.MSELoss()
    elif loss_function == "ce":
        loss_fn = nn.CrossEntropyLoss()
    elif loss_function == "bce":
        loss_fn = nn.BCELoss()
    elif loss_function == "bcewll":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise Exception("Loss function not known.")
    
    return loss_fn

def get_scheduler(model, optimizer, scheduler_name, scheduler_step_size, scheduler_gamma, max_epochs):
    #initialize learning rate scheduler
    if scheduler_name == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    elif scheduler_name == "cosineannealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif scheduler_name == 'reduceonplateau':
        scheduler = torch.optim.lr_scheduler.REDUCELRONPLATEAU(optimizer, mode='min', factor=scheduler_gamma, patience=10,
                                                               threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0)
    elif scheduler_name == None:
        return None     

    else:
        raise Exception("Scheduler not known.")
    return scheduler    

class EarlyStopper:
    """
    Stop training when the best validation error is at least N epochs (patience) past
    """
    
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')

    def early_stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class RectifiedTanh(nn.Module):
    def __init__(self):
        super(RectifiedTanh, self).__init__()

    def forward(self, x):
        return torch.max(torch.tanh(x), torch.zeros_like(x))

class RNN(nn.Module):
    def __init__(self, dims, noise_std=0., dt=0.5, 
                 nonlinearity='tanh', readout_nonlinearity='id',
                 g=None, g_in=1, wi_init=None, wrec_init=None, wo_init=None, brec_init=None, bo_init=None,
                 h0_init=None, hidden_initial_variance=0., oth_init=None,
                 train_wi=True, train_wrec=True, train_wo=True, train_brec=True, train_bo=True, train_h0=True,
                 ML_RNN=True, map_output_to_hidden=False, input_nonlinearity=None, save_inputs=False):
        """
        :param dims: list = [input_size, hidden_size, output_size]
        :param noise_std: float
        :param dt: float, integration time step
        :param nonlinearity: str, nonlinearity. Choose 'tanh' or 'id'
        :param readout_nonlinearity: str, nonlinearity. Choose 'tanh' or 'id'
        :param g: float, std of gaussian distribution for initialization
        :param wi_init: torch tensor of shape (input_dim, hidden_size)
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param wrec_init: torch tensor of shape (hidden_size, hidden_size)
        :param brec_init: torch tensor of shape (hidden_size)
        :param h0_init: torch tensor of shape (hidden_size)
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_brec: bool
        :param train_h0: bool
        :param ML_RNN: bool; whether forward pass is ML convention f(Wr)
        """
        super(RNN, self).__init__()
        self.dims = dims
        input_size, hidden_size, output_size = dims
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.hidden_initial_variance = hidden_initial_variance
        self.dt = dt
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_brec = train_brec
        self.train_h0 = train_h0
        self.ML_RNN = ML_RNN
        self.save_inputs = save_inputs
        self.map_output_to_hidden = map_output_to_hidden #oth
        
        # Either set g or choose initial parameters. Otherwise, there's a conflict!
        assert (g is not None) or (wrec_init is not None), "Choose g or initial wrec!"

        self.g = g
        
        # Nonlinearity
        if nonlinearity == 'tanh':
            self.nonlinearity = torch.tanh
        elif nonlinearity == 'id':
            self.nonlinearity = lambda x: x
            if g is not None:
                if g > 1:
                    warn("g > 1. For a linear network, we need stable dynamics!")
        elif nonlinearity.lower() == 'relu':
            self.nonlinearity = nn.ReLU()
        elif nonlinearity.lower() == 'rect_tanh':
            # if self.noise_std!=0.:
            # self.nonlinearity = nn.ReLU(torch.tanh)                 ##works with noise
            # self.nonlinearity = lambda x: 1.*(torch.tanh(x)>0)   #works with ml_rnn False
            self.nonlinearity = RectifiedTanh()

        elif nonlinearity.lower() == 'talu':
            self.nonlinearity = lambda x: torch.tanh(x) if x<0 else x
        elif nonlinearity == 'softplus':
            softplus_scale = 1 # Note that scale 1 is quite far from relu
            self.nonlinearity = lambda x: torch.log(1. + torch.exp(softplus_scale * x)) / softplus_scale
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = lambda x: 1 / (1 + torch.exp(-x))
        elif type(nonlinearity) == str:
            raise NotImplementedError("Nonlinearity not yet implemented.")
        else:
            self.nonlinearity = nonlinearity
            
            
        # Readout nonlinearity
        if readout_nonlinearity is None:
            # Same as recurrent nonlinearity
            self.readout_nonlinearity = self.nonlinearity
        elif readout_nonlinearity == 'tanh':
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
            
        # Input mapping nonlinearity
        if input_nonlinearity is None or input_nonlinearity == 'id':
            self.input_nonlinearity = lambda x: x
        elif input_nonlinearity == 'recurrent':
            self.input_nonlinearity = self.nonlinearity
            

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        if self.ML_RNN=='noorman':
            self.wi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_wi:
            self.wi.requires_grad = False
            
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_wrec:
            self.wrec.requires_grad = False
        self.brec = nn.Parameter(torch.Tensor(hidden_size))
        if not train_brec:
            self.brec.requires_grad = False
            
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        if not train_wo:
            self.wo.requires_grad = False
        self.bo = nn.Parameter(torch.Tensor(output_size))
        if not train_bo:
                self.bo.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_(std=g_in /np.sqrt(hidden_size))
            else:
                if type(wi_init) == np.ndarray:
                    wi_init = torch.from_numpy(wi_init)
                self.wi.copy_(wi_init)
            if wrec_init is None:
                self.wrec.normal_(std=g / np.sqrt(hidden_size))
            else:
                if type(wrec_init) == np.ndarray:
                    wrec_init = torch.from_numpy(wrec_init)
                self.wrec.copy_(wrec_init)
            if wo_init is None:
                self.wo.normal_(std=1 / np.sqrt(hidden_size))
            else:
                if type(wo_init) == np.ndarray:
                    wo_init = torch.from_numpy(wo_init)
                self.wo.copy_(wo_init)
            if brec_init is None:
                self.brec.zero_()
                torch.nn.init.uniform_(self.brec, a=-np.sqrt(hidden_size), b=np.sqrt(hidden_size))
            else:
                if type(brec_init) == np.ndarray:
                    brec_init = torch.from_numpy(brec_init)
                self.brec.copy_(brec_init)
            if bo_init is None:
                self.bo.zero_()
            else:
                if type(bo_init) == np.ndarray:
                    bo_init = torch.from_numpy(bo_init)
                self.bo.copy_(bo_init)
            if h0_init is None:
                torch.nn.init.uniform_(self.h0, a=-1, b=1)
            else:
                if type(h0_init) == np.ndarray:
                    h0_init = torch.from_numpy(h0_init)
                    self.h0.copy_(h0_init)
                
        if map_output_to_hidden:
            self.h0.requires_grad = False
            self.output_to_hidden = nn.Parameter(torch.Tensor(output_size, hidden_size))

            if oth_init is None:
                with torch.no_grad():
                    self.output_to_hidden.normal_(std=1 / np.sqrt(hidden_size))
            else:
                with torch.no_grad():
                    oth_init = torch.from_numpy(oth_init)
                    self.output_to_hidden.copy_(oth_init)
            # self.oth_nonlinearity = nonlinearity
            self.oth_nonlinearity = lambda x: x

        
            
    def forward(self, input, return_dynamics=False, h_init=None, target=None):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: bool
        :return: if return_dynamics=False, output tensor of shape (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        
        # assert not self.map_output_to_hidden or target
        
        batch_size = input.shape[0]
        seq_len = input.shape[1]

        if self.map_output_to_hidden:
            h_init_torch = nn.Parameter(torch.Tensor(batch_size, self.hidden_size))
            h_init_torch.requires_grad = False
            h_init_torch = target[:,0,:].matmul(self.output_to_hidden)
            h_init_torch = self.input_nonlinearity(h_init_torch)
            with torch.no_grad():
                h = h_init_torch
                
        elif type(h_init) == np.ndarray or isinstance(h_init, torch.Tensor):
            h_init_torch = nn.Parameter(torch.Tensor(batch_size, self.hidden_size))
            h_init_torch.requires_grad = False
            # Initialize parameters 
            with torch.no_grad():
                if type(h_init) == np.ndarray:
                    h = h_init_torch.copy_(torch.from_numpy(h_init))
                else:
                    h = h_init_torch.copy_(h_init)
                
        elif h_init == 'random':
            h = torch.normal(mean=0, std=self.hidden_initial_variance, size=(1, batch_size, self.hidden_size)).to(self.wrec.device)
            self.h0.requires_grad = False

        elif h_init == 'self_h':
             h = self.h0
                
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.wrec.device)

        # simulation loop
        for i in range(seq_len):

            if self.ML_RNN:
                rec_input = self.nonlinearity(
                    h.matmul(self.wrec.t()) 
                    + input[:, i, :].matmul(self.wi)
                    + self.brec)
                h = ((1 - self.dt) * h 
                     + self.dt * rec_input
                     + np.sqrt(self.dt) * self.noise_std * noise[:, i, :])
                out_i = self.readout_nonlinearity(h.matmul(self.wo))+self.bo
                
            else:
                rec_input = (
                    self.nonlinearity(h).matmul(self.wrec.t()) 
                    + input[:, i, :].matmul(self.wi) 
                    + self.brec)
                h = ((1 - self.dt) * h 
                     + self.dt * rec_input
                     + np.sqrt(self.dt) * self.noise_std * noise[:, i, :])
                out_i = self.readout_nonlinearity(h).matmul(self.wo)+self.bo

            output[:, i, :] = out_i

            if return_dynamics:
                trajectories[:, i, :] = h

        if not return_dynamics:
            return output
        else:
            return output, trajectories


def get_weights_during(net, wis=None, wrecs=None, wos=None, brecs=None, h0s=None, oths=None):

    # Final weights
    wi_last = net.wi.cpu().detach().numpy().copy()
    wrec_last = net.wrec.cpu().detach().numpy().copy()
    wo_last = net.wo.cpu().detach().numpy().copy()
    brec_last = net.brec.cpu().detach().numpy().copy()
    h0_last = net.h0.cpu().detach().numpy().copy()
    weights_last = [wi_last, wrec_last, wo_last, brec_last, h0_last]
    if net.map_output_to_hidden:
        oth_last = net.output_to_hidden.cpu().detach().numpy()
        weights_last = [wi_last, wrec_last, wo_last, brec_last, h0_last, oth_last]
    
    # Weights throughout training: 
    weights_train = {}
    if net.train_wi:
        weights_train["wi"] = wis
    if net.train_wrec:
        weights_train["wrec"] = wrecs
    if net.train_wo:
        weights_train["wo"] = wos
    if net.train_brec:
        weights_train["brec"] = brecs
    if net.train_h0:
        weights_train["h0"] = h0s
    if net.map_output_to_hidden:
        weights_train["oths"] = oths
        
    return weights_train, weights_last

def train(net, task=None, data=None, n_epochs=10, batch_size=32, learning_rate=1e-2, clip_gradient=None, cuda=False, record_step=1,
          h_init=None, hidden_initial_variance=0,
          loss_function='mse_loss_masked', final_loss=True, last_mses=None, act_reg_lambda=0.,
          optimizer='sgd', momentum=0, weight_decay=.0, adam_betas=(0.9, 0.999), adam_eps=1e-8, #optimizers 
          scheduler=None, scheduler_step_size=100, scheduler_gamma=0.3, 
          stop_patience=10, stop_min_delta=0, 
          verbose=True, experiment_folder=None):
    """
    Train a network
    :param net: nn.Module
    :param task: function; generates input, target, mask for a single batch
    :param n_epochs: int
    :param batch_size: int
    :param learning_rate: float
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param cuda: bool
    :param record_step: int; record weights after these steps
    :return: res
    """
    assert (task is not None) or (data is not None), "Choose a task or a dataset!"
 
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
    
    h_init_type = h_init
    
    # Optimizer
    optimizer = get_optimizer(net, optimizer, learning_rate, weight_decay, momentum, adam_betas, adam_eps)
    
    scheduler = get_scheduler(net, optimizer, scheduler, scheduler_step_size, scheduler_gamma, n_epochs)

    loss_function = get_loss_function(net, loss_function)
    
    early_stopper = EarlyStopper(patience=stop_patience, min_delta=stop_min_delta)
    
    if data:
        train_input, train_target, train_mask = data['train_input'], data['train_target'], data['train_mask']
        validation_input, validation_target, validation_mask = data['validation_input'], data['validation_target'], data['validation_mask']
        # Convert training data to pytorch tensors
        validation_input = torch.from_numpy(validation_input)
        validation_target = torch.from_numpy(validation_target)
        validation_mask = torch.from_numpy(validation_mask)
        # Allocate
        validation_input = validation_input.to(device=device).float() 
        validation_target = validation_target.to(device=device).float() 
        validation_mask = validation_mask.to(device=device).float() 
        dataset = TensorDataset(torch.tensor(train_input, dtype=torch.float, device=device),
                                torch.tensor(train_target, dtype=torch.float, device=device),
                                torch.tensor(train_mask, dtype=torch.float, device=device))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
    # Save initial weights
    wi_init = net.wi.cpu().detach().numpy().copy()
    wrec_init = net.wrec.cpu().detach().numpy().copy()
    wo_init = net.wo.cpu().detach().numpy().copy()
    brec_init = net.brec.cpu().detach().numpy().copy()
    h0_init = net.h0.cpu().detach().numpy().copy()
    weights_init = [wi_init, wrec_init, wo_init, brec_init, h0_init]
    
    # Record
    dim_rec = net.hidden_size
    dim_in = net.input_size
    dim_out = net.output_size
    n_rec_epochs = n_epochs // record_step
    
    losses = np.zeros((n_epochs), dtype=np.float32)
    validation_losses = np.zeros((n_epochs), dtype=np.float32)
    gradient_norm_sqs = np.zeros((n_epochs), dtype=np.float32)
    epochs = np.zeros((n_epochs))
    rec_epochs = np.zeros((n_rec_epochs))
    if net.train_wi:
        wis = np.zeros((n_rec_epochs, dim_in, dim_rec), dtype=np.float32)
    if net.train_wrec:
        wrecs = np.zeros((n_rec_epochs, dim_rec, dim_rec), dtype=np.float32)
    if net.train_wo:
        wos = np.zeros((n_rec_epochs, dim_rec, dim_out), dtype=np.float32)
    if net.train_brec:
        brecs = np.zeros((n_rec_epochs, dim_rec), dtype=np.float32)
    if net.train_h0:
        h0s = np.zeros((n_rec_epochs, dim_rec), dtype=np.float32)
    if net.map_output_to_hidden:
        oths = np.zeros((n_rec_epochs, dim_out, dim_rec), dtype=np.float32)
    else: 
        oths=None
    if net.save_inputs:
        _input, _target, _mask = task(batch_size)
        all_inputs = np.zeros((n_rec_epochs, batch_size, _input.shape[1], dim_in), dtype=np.float32)
        all_targets = np.zeros((n_rec_epochs, batch_size, _target.shape[1], dim_out), dtype=np.float32)
        all_trajectories = np.zeros((n_rec_epochs, batch_size, _target.shape[1], dim_rec), dtype=np.float32)
        all_outputs = np.zeros((n_rec_epochs, batch_size, _target.shape[1], dim_out), dtype=np.float32)


    time0 = time.time()
    if verbose:
        print("Training...")
    for i in range(n_epochs):
        # Save weights (before update)
        if i % record_step == 0:
            k = i // record_step
            rec_epochs[k] = i
            if net.train_wi:
                wis[k] = net.wi.cpu().detach().numpy()
            if net.train_wrec:
                wrecs[k] = net.wrec.cpu().detach().numpy()
            if net.train_wo:
                wos[k] = net.wo.cpu().detach().numpy()
            if net.train_brec:
                brecs[k] = net.brec.cpu().detach().numpy()
            if net.train_h0:
                h0s[k] = net.h0.cpu().detach().numpy()
            if net.map_output_to_hidden:
                oths[k] = net.output_to_hidden.cpu().detach().numpy()
        
        if not data:
            # Generate batch
            if  i!=0 and h_init_type=='prev_last':
                h_init=trajectories[:,-1,:]
            elif h_init_type=='prev_last':
                h_init='random'
                
            _input, _target, _mask = task(batch_size)

            # Convert training data to pytorch tensors
            _input = torch.from_numpy(_input)
            _target = torch.from_numpy(_target)
            # Allocate
            input = _input.to(device=device).float() 
            target = _target.to(device=device).float() 
        
            optimizer.zero_grad()
            output, trajectories = net(input, h_init=h_init, return_dynamics=True, target=target)
            
            #apply mask after output
            
            
            if np.any(_mask):
                _mask = torch.from_numpy(_mask)
                mask = _mask.to(device=device).float() 
                loss = loss_function(output, target, mask)
            else: 
                # print(output)
                # loss = loss_function(output.view(-1), target.view(-1))
                loss = loss_function(output[:,-1,:], target[:,-1,:])
                # print("L", loss)
                # loss = loss_function(output[...,0], target[...,0])
                # loss += loss_function(output[...,1], target[...,1])

                # for t_id in range(x.shape[1]):
                    
            if net.save_inputs:
                all_inputs[k] = _input
                all_targets[k] = _target
                all_trajectories[k] = trajectories.detach().numpy()
                all_outputs[k] = output.detach().numpy()
                        
            act_reg = 0.
            if act_reg_lambda != 0.: 
                act_reg += torch.mean(torch.linalg.norm(trajectories, dim=2))
                # act_reg /= trajectories.shape[1]
                loss += act_reg_lambda*act_reg
            
            # Gradient descent
            loss.backward()
            if clip_gradient is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
            # [print(p.grad) for p in net.parameters() if p.requires_grad]
            gradient_norm_sq = sum([(p.grad ** 2).sum() for p in net.parameters() if p.requires_grad and p.grad!=None])
            
            if any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any() for p in net.parameters() if p.grad is not None):
                if verbose:
                    print("NaN or Inf detected in gradients. Stopping training.")
                
                break
            
            # Update weights
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            # Important to prevent memory leaks:
            loss.detach()
            output.detach()
        else:
            for data_idx, (input, target, mask) in enumerate(data_loader):
            
                optimizer.zero_grad()
                output = net(input, h_init=h_init)
                loss = loss_function(output, target, mask)
                
                # Gradient descent
                loss.backward()
                if clip_gradient is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
                gradient_norm_sq = sum([(p.grad ** 2).sum() for p in net.parameters() if p.requires_grad])
                
                # Update weights
                optimizer.step()
                
                if scheduler:
                    scheduler.step()
                
                # These 2 lines important to prevent memory leaks
                loss.detach()
                output.detach()
        
        if data:
            output = net(input, h_init=h_init)
            loss = loss_function(output, target, mask)
            #test on validation dataset
            output = net(validation_input, h_init=h_init)
            validation_loss = loss_function(output, validation_target, validation_mask)
            validation_losses[i] = validation_loss.item()
            
        if early_stopper.early_stop(loss.item()):             
            break

        # Save
        epochs[i] = i
        losses[i] = loss.item()
        gradient_norm_sqs[i] = gradient_norm_sq
        
        if verbose:
            if data:
                print("epoch %d / %d:  loss=%.6f, run.loss=%.6f, val.loss=%.6f \n" % (i+1, n_epochs, losses[i], np.mean(losses[:i]), validation_loss))
            else:
                print("epoch %d / %d:  loss=%.6f, run.loss=%.6f \n" % (i+1, n_epochs, losses[i], np.mean(losses[:i])))
                
        if i % record_step == 0 and i!=0: 
            gradient_norms = np.sqrt(gradient_norm_sqs)
            weights_train, weights_last = get_weights_during(net, wis, wrecs, wos, brecs, h0s, oths)            
            res = [losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs]

            with open(experiment_folder + '/res_weights.pickle', 'wb') as handle:
                pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    if verbose:
        print("\nDone. Training took %.1f sec." % (time.time() - time0))
    
    # Obtain gradient norm
    gradient_norms = np.sqrt(gradient_norm_sqs)
    
    weights_train, weights_last = get_weights_during(net, wis, wrecs, wos, brecs, h0s, oths)
    
    res = [losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs]
    if net.save_inputs:
        res_dict = {"losses":losses, "validation_losses":validation_losses, "gradient_norms":gradient_norms,
                    "weights_init":weights_init, "weights_last":weights_last, "weights_train":weights_train,
                    "epochs":epochs, "rec_epochs":rec_epochs,
                    "all_inputs":all_inputs, "all_targets":all_targets, "all_outputs":all_outputs, "all_trajectories":all_trajectories}
    else:
        res_dict = {"losses":losses, "validation_losses":validation_losses, "gradient_norms":gradient_norms,
                    "weights_init":weights_init, "weights_last":weights_last, "weights_train":weights_train,
                    "epochs":epochs, "rec_epochs":rec_epochs}

    return res, res_dict

def run_net(net, task, batch_size=32, return_dynamics=False, h_init=None):
    # Generate batch
    input, target, mask = task(batch_size)
    # Convert training data to pytorch tensors
    input = torch.from_numpy(input).float() 
    target = torch.from_numpy(target).float() 
    mask = torch.from_numpy(mask).float() 

    with torch.no_grad():
        # Run dynamics
        if return_dynamics:
            output, trajectories = net(input, return_dynamics, h_init=h_init, target=target)
        else:
            output = net(input, h_init=h_init, target=target)
        loss = mse_loss_masked(output, target, mask)
    res = [input, target, mask, output, loss]
    if return_dynamics:
        res.append(trajectories)
    res = [r.numpy() for r in res]
    return res


def run_net_dataset(net, input, target, mask, batch_size=32, return_dynamics=False, h_init=None):
    with torch.no_grad():
        # Run dynamics
        if return_dynamics:
            output, trajectories = net(input, return_dynamics, h_init=h_init)
        else:
            output = net(input, h_init=h_init)
        loss = mse_loss_masked(output, target, mask)
    res = [input, target, mask, output, loss]
    if return_dynamics:
        res.append(trajectories)
    res = [r.numpy() for r in res]
    return res


