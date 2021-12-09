#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 09:07:34 2021

@author: oster
"""

import numpy as np
import scipy
import set_dynamics
import ode
from misc import  legendre_measures, legendre_measures_grad  #, hermite_measures
import warnings
warnings.filterwarnings('error')
ode=ode.Ode()
A = np.load('A.npy')
B = np.load('B.npy')
Q_discr =np.load('Q.npy')
R_discr =np.load('R.npy')
load_me = np.load('data.npy')
t_vec = np.load('t_vec_p.npy')
tau_value = t_vec[1]-t_vec[0]
lambd = load_me[0]
a = -load_me[2]
b = -a
tau = load_me[3]
R_inv =np.load('R_inv.npy')
Q = Q_discr/tau
R = lambd*R_discr/tau
tau_value_func = t_vec[1]-t_vec[0]

def calc_tilde_r(t,x,vlist):
    rew=[]
    for s in np.linspace(t,t+tau_value_func,int(tau_value_func/tau)):
        u = ode.calc_u(s,x,gradV(x,vlist[-1]))
        rew.append(ode.calc_reward(s, x, u))
        
        x=ode.step(s,x,u)
    u = ode.calc_u(t+tau_value_func,x,gradV(x,vlist[-1]))
    rew.append(ode.calc_reward(t+tau_value_func, x, u))    
    ret = scipy.integrate.trapz(np.array(rew),axis=0)+V(x,vlist[-2])
    print('ret.shape in cal_tilde_r', ret.shape)
    return ret

def calc_total_reward(x,steps, vlist):
    rew=[]
    u_opt = np.zeros([1,len(steps)])
    x_opt = np.zeros([x.shape[0],len(steps)])
    for i in range(len(steps)-1):
        t = steps[i]
        j = t_to_ind(t)
        print('j',j,'len(vlist_)',len(vlist))
        #for s in np.linspace(t,t+tau_value_func,int(tau_value_func/tau)):
            #print('len(vlist[:-1])',i,len(vlist[:i+1]))
        #u = calc_u(s,x,vlist[:len(vlist)-1-i])
        x_opt[:,i] = x[:,0]
        u  = ode.calc_u(t,x,gradV(x,vlist[-1]))    
        rew.append(ode.calc_reward(t, x, u))
        x=ode.step(t,x,u)
        u_opt[:,i] = u[:,0] 
        if i%int(tau_value_func/tau) ==9: 
            vlist.pop()
    # for i in range(1,len(vlist)-1):
    #     t = t_vec[i]
    #     for s in np.linspace(t,t+tau_value_func,int(tau_value_func/tau)):
    #         u = calc_u(s,x,vlist[:-i])
    #         rew.append(calc_reward(s, x, u))
    #         x=step(s,x,u)
    x_opt[:,-1 ] = x[:,0]
    #x_opt = np.array(x_opt)
    x_wub = x_opt[:,-1]
    print('x_opt.shape',x_opt.shape)
    u  = ode.calc_u(steps[-1],x,gradV(x,vlist[-1]))
    u_opt[:,i] = u[:,0] 
    rew.append(ode.calc_reward(steps[-1], x, u))
    print('ode.calc_end_reward(0,x), V(x,vlist[0])',ode.calc_end_reward(0,x), V(x,vlist[0]))
    ret = scipy.integrate.trapz(np.array(rew),axis=0)+ode.calc_end_reward(0, x)#V(x,vlist[0])
    #u_opt  = np.reshape(np.array(u_opt),(u_opt[0].shape[0],len(u_opt)))
    #x_opt  = np.reshape(np.array(x_opt),(x_opt[0].shape[0],len(x_opt)))
    print('ret u, x shape', ret.shape, u_opt.shape,x_opt.shape)
    return ret,u_opt,x_opt


def t_to_ind( t):
        # print('t, t/self.tau, int(t/self.tau, int(np.round(t/self.tau))', t, t/self.tau, int(t/self.tau), int(np.round(t/self.tau, 8)))
    return int(np.round(t/tau_value_func,8))
# def calc_u( t, x, vlist):
#    u_mat = np.tensordot(gradV(x,vlist[-1]), B, axes=((0),(0)))
#    return -R_inv @ u_mat.T / 2
# def step( t, x, u):
#     return step_rk4(t, x, u, rhs_schloegl)


# def step_rk4( t, x, u, rhs):
#     k1 = tau * rhs(t, x, u)
#     k2 = tau * rhs(t+tau/2, x + k1/2, u)
#     k3 = tau * rhs(t+tau/2, x + k2/2, u)
#     k4 = tau * rhs(t+tau, x + k3, u)
#     return x + 1/6*(k1 + 2*k2 + 2*k3 + k4)


# def rhs_schloegl( t, x, u):
#     return f(t, x) +g(t, x) @ u
# def f( t, x):
#     return A@x+NL(t, x)


# def NL( t, x):
#     return -x**3


# def g( t, x):    
#     return B

# def q( t, x):
#     return np.einsum('il,ik,kl->l', x,Q_discr,x)
# def r( t, u):
#     return np.einsum('il,ik,kl->l', u,R_discr,u)

# def calc_reward(t, x, u):
#     return q(t, x) +  r(t, u)
# def calc_end_reward( t, x):
        
#         #print("caLC_ENDREWARD", self.G(x))
#         #return self.G(x)
#     if len(x.shape) == 1:
#         return x.T @ Q_discr @ x/tau
#     else:
#         return np.einsum('il,ik,kl->l', x,Q_discr,x)/tau

def V(x,v):
    order,samples = x.shape
    deg = v.dimensions[0]-1
    measure = legendre_measures(x.T,deg,a,b)
    measure =  np.concatenate([measure, np.ones((1,samples,deg+1))], axis=0)
    res = v.evaluate(measure)
    assert res.shape ==  (samples,)
    return res

def gradV(x,v):
    order,samples = x.shape
    deg = v.dimensions[0]-1
    measure_grad = legendre_measures_grad(x.T,deg,a,b)
    measure_grad = [np.concatenate([measure_grad[k], np.ones((1,samples,deg+1))], axis=0) for k in range(order)] 
    res = np.array([v.evaluate(m) for m in measure_grad])
    assert res.shape ==  (order,samples) 
    return res

