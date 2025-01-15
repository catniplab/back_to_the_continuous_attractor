# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:37:38 2024

@author: 
"""

import numpy as np
from scipy.integrate import solve_ivp

def ReLU(x):
    return np.where(x<0,0,x)

def ReTanh(x):
    return np.where(x<0,0,np.tanh(x))

def talu(x):
    y = np.where(x<0,np.tanh(x),x)
    return y


def lu_step(x, W, b):
    return x*W+b

def relu_step(x, W, b):
    res = np.array(np.dot(W,x)+b)
    res[res < 0] = 0
    return res
 
def relu_step_input(x, W, b, W_ih=None, I=None):
    if I:
        res = np.array(np.dot(W,x) + b + np.dot(W_ih, I))
    else:
        res = np.array(np.dot(W,x) + b)
    res[res < 0] = 0
    return res

def talu_step(x, wrec, brec, dt):
    act = np.dot(wrec, x) + brec
    rec_input = np.where(act<0,np.tanh(act),act)
    dx =  - dt * x  + dt * rec_input
    return dx

def rnn_step(x, wrec, brec, dt, nonlinearity):
    act = np.dot(wrec, x) + brec
    rec_input = nonlinearity(act)
    dx =  - dt * x  + dt * rec_input
    return dx

def rect_tanh_step(x, wrec, brec, dt):
    tanh_act = np.tanh(np.dot(wrec, x) + brec)
    rec_input = np.where(tanh_act>0,tanh_act,0)
    dx =  - dt * x  + dt * rec_input
    return dx

def talu_step(x, wrec, brec, dt):
    act = np.dot(wrec, x) + brec
    rec_input = np.where(act<0,np.tanh(act),act)
    dx =  - dt * x  + dt * rec_input
    return dx

def relu_ode(t,x,W,b,tau,mlrnn=True):

    if mlrnn:
        return (-x + ReLU(np.dot(W,x)+b))/tau
    else:
        return (-x + np.dot(W,ReLU(x))+b)/tau

def tanh_ode(t,x,W,b,tau,mlrnn=True):
    if mlrnn:
        return (-x + np.tanh(np.dot(W,x)+b))/tau
    else:
        return (-x + np.dot(W,np.tanh(x))+b)/tau

def recttanh_ode(t,x,W,b,tau,mlrnn=True):
    if mlrnn:
        return (-x + ReTanh(np.dot(W,x)+b))/tau
    else:
        return (-x + np.dot(W,ReTanh(x))+b)/tau

def get_rnn_ode(nonlinearity):
    if nonlinearity == 'tanh':
        return tanh_ode
    elif nonlinearity == 'relu':
        return relu_ode
    elif nonlinearity == 'rect_tanh':
        return recttanh_ode
    
    
def rnn_speed_function(x, wrec, brec, dt, nolinearity):
    return np.linalg.norm(rnn_step(x, wrec, brec, dt, nolinearity))**2

def rnn_speed_function_in_outputspace(x, wrec, brec, wo, dt, nonlinearity):
    return np.linalg.norm(np.dot(wo.T, rnn_step(x, wrec, brec, dt, nonlinearity)))**2


######numerical integration
def simulate_network(W, b, nonlinearity_ode=relu_ode, y0=None, maxT=25, tsteps=501, tau=1, mlrnn=True):
    
    N = W.shape[0]
    if not np.any(y0):
        y0 = np.random.uniform(0,1,N)
    t = np.linspace(0, maxT, tsteps)
    sol = solve_ivp(nonlinearity_ode, y0=y0,  t_span=[0,maxT],
                    args=tuple([W, b, tau, mlrnn]),
                    dense_output=True)
    return sol.sol(t)


def simulate_network_ntimes(Nsims, W, b, nonlinearity_ode=relu_ode, mlrnn=True,
                            y0s=None, y0_dist="uniform", #todo: y0_dist=
                            maxT=25, tsteps=501, tau=1):
    t = np.linspace(0, maxT, tsteps)
    N = W.shape[0]
    sols = np.zeros((Nsims, tsteps, N))
    for ni in range(Nsims):
        if not np.any(y0s):
            if y0_dist=='uniform':
                y0 = np.random.uniform(-1,1,N)
            else:
                y0 = np.random.normal(0,1,N)
        else:
            y0 = y0s[ni]
        sol = solve_ivp(nonlinearity_ode, y0=y0,  t_span=[0,maxT],
                        args=tuple([W, b, tau, mlrnn]),
                        dense_output=True)
        sols[ni,...] = sol.sol(t).T.copy()
    return sols



#simulate_from initial values
def simulate_from_y0s(y0s, W, b, tau=1, 
                   maxT = 25, tsteps=501):

    N = W.shape[0]
    t = np.linspace(0, maxT, tsteps)
    sols = np.zeros((y0s.shape[1], t.shape[0], N))
    for yi,y0 in enumerate(y0s.T):
        sol = solve_ivp(relu_ode, y0=y0, t_span=[0,maxT],
                        args=tuple([W, b, tau]),
                        dense_output=True)
        sols[yi,...] = sol.sol(t).T

    return sols

##############linearization
#Jacobians

def relu_jacobian(W,b,tau,x):
    return (-np.eye(W.shape[0]) + np.multiply(W, np.where(np.dot(W,x)+b>0,1,0)))/tau

def tanh_jacobian(W,b,tau,x,mlrnn=True):
    if mlrnn:
        #return (-np.eye(W.shape[0]) + np.multiply(W,1/np.cosh(np.dot(W,x)+b)**2))/tau
        dtanh = 1 - np.tanh(np.dot(W, x) + b) ** 2
        df_dx = -np.eye(len(x))/tau + (np.dot(np.diag(dtanh), W))/tau
        return df_dx
    else:
        return (-np.eye(W.shape[0]) + np.multiply(W,1/np.cosh(x)**2))/tau


def recttanh_jacobian_point(W,b,tau,x):
    #b is unused, but there for consistency with relu jac
    return (-np.eye(W.shape[0]) + np.multiply(np.multiply(W, np.where(np.dot(W,np.tanh(x))+b>0,1,0)),  np.multiply(W,1/np.cosh(x)**2)))/tau




#for trajectory
def linear_jacobian(t,W,b,tau,x_solved):
    return W/tau

def relu_jacobian_sequence(t,W,b,tau,x_solved):
    return (-np.eye(W.shape[0]) + np.multiply(W, np.where(np.dot(W,x_solved[t])+b>0,1,0)))/tau

def tanh_jacobian_sequence(t,W,b,tau,x_solved, mlrnn=True):
    
    if mlrnn:
        return (-np.eye(W.shape[0]) + np.multiply(W,1/np.cosh(np.dot(W,x_solved[t])+b)**2))/tau
    else:
        return (-np.eye(W.shape[0]) + np.multiply(W,1/np.cosh(x_solved[t])**2))/tau



#########finding fixed points
def newton_method(x0, system, jacobian, W, b, tau, mlrnn, tol=1e-6, max_iter=100):
    # Implement the Newton method
    x = x0
    for _ in range(max_iter):
        dx = np.linalg.solve(jacobian(W,b,tau,x,mlrnn), system(0,x,W,b,tau,mlrnn))
        x -= dx
        if np.linalg.norm(dx) < tol:
            break
    return x


