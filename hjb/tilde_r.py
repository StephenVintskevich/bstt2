#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 09:07:34 2021

@author: oster
"""

import numpy as np
import scipy
import set_dynamics
from misc import  legendre_measures, legendre_measures_grad  #, hermite_measures
import warnings
warnings.filterwarnings('error')

A = np.load('A.npy')
B = np.load('B.npy')
Q_discr =np.load('Q.npy')
R_discr =np.load('R.npy')
load_me = np.load('data.npy')
t_vec = np.load('t_vec_p.npy')
tau_value = t_vec[1]-t_vec[0]
lambd = load_me[0]
interval_half = load_me[2]
tau = load_me[3]
R_inv =np.load('R_inv.npy')
Q = Q_discr/tau
R = lambd*R_discr/tau
tau_value_func = t_vec[1]-t_vec[0]

def calc_tilde_r(t,x,vlist):
    rew=[]
    for s in np.linspace(t,t+tau_value_func,int(tau_value_func/tau)):
        u = calc_u(s,x,vlist)
        rew.append(calc_reward(s, x, u))
        x=step(s,x,u)
    ret = scipy.integrate.trapz(np.array(rew),axis=0)+V(x,vlist[-2])
    return ret

def calc_total_reward(x,steps, vlist):
    rew=[]
    u_opt = []
    for i in range(len(steps)):
        t = steps[i]
        j = t_to_ind(t)
        print('j',j,'len(vlist_)',len(vlist))
        #for s in np.linspace(t,t+tau_value_func,int(tau_value_func/tau)):
            #print('len(vlist[:-1])',i,len(vlist[:i+1]))
        #u = calc_u(s,x,vlist[:len(vlist)-1-i])
        u  = calc_u(t,x,vlist)    
        rew.append(calc_reward(t, x, u))
        x=step(t,x,u)
        u_opt.append(u)
        if i%int(tau_value_func/tau) ==9: vlist.pop()
    for i in range(1,len(vlist)-1):
        t = t_vec[i]
        for s in np.linspace(t,t+tau_value_func,int(tau_value_func/tau)):
            u = calc_u(s,x,vlist[:-i])
            rew.append(calc_reward(s, x, u))
            x=step(s,x,u)
    ret = scipy.integrate.trapz(np.array(rew),axis=0)+V(x,vlist[0])
    u_opt  = np.reshape(np.array(u_opt),(u_opt[0].shape[0],len(u_opt)))
    return ret,u_opt


def t_to_ind( t):
        # print('t, t/self.tau, int(t/self.tau, int(np.round(t/self.tau))', t, t/self.tau, int(t/self.tau), int(np.round(t/self.tau, 8)))
    return int(np.floor(t/tau_value_func))
def calc_u( t, x, vlist):
   u_mat = np.tensordot(gradV(x,vlist[-1]), B, axes=((0),(0)))
   return -R_inv @ u_mat.T / 2
def step( t, x, u):
    return step_rk4(t, x, u, rhs_schloegl)


def step_rk4( t, x, u, rhs):
    k1 = tau * rhs(t, x, u)
    k2 = tau * rhs(t+tau/2, x + k1/2, u)
    k3 = tau * rhs(t+tau/2, x + k2/2, u)
    k4 = tau * rhs(t+tau, x + k3, u)
    return x + 1/6*(k1 + 2*k2 + 2*k3 + k4)


def rhs_schloegl( t, x, u):
    return f(t, x) +g(t, x) @ u
def f( t, x):
    return A@x+NL(t, x)


def NL( t, x):
    return -x**3


def g( t, x):    
    return B

def q( t, x):
    return np.einsum('il,ik,kl->l', x,Q_discr,x)
def r( t, u):
    return np.einsum('il,ik,kl->l', u,R_discr,u)

def calc_reward(t, x, u):
    return q(t, x) +  r(t, u)


def V(x,v):
    order,samples = x.shape
    deg = v.dimensions[0]-1
    measure = legendre_measures(x.T,deg)
    measure =  np.concatenate([measure, np.ones((1,samples,deg+1))], axis=0)
    res = v.evaluate(measure)
    assert res.shape ==  (samples,)
    return res

def gradV(x,v):
    order,samples = x.shape
    deg = v.dimensions[0]-1
    measure_grad = legendre_measures_grad(x.T,deg)
    measure_grad = [np.concatenate([measure_grad[k], np.ones((1,samples,deg+1))], axis=0) for k in range(order)] 
    res = np.array([v.evaluate(m) for m in measure_grad])
    assert res.shape ==  (order,samples) 
    return res

