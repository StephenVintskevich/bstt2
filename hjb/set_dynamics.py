#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:02:03 2019

@author: sallandt

Builds system matrices and saves them. Also calculates an initial control. xerus dependencies can be deleted.
"""
import numpy as np
from scipy import linalg as la
import pickle

num_valuefunctions = 31#301
T = 0.3
t_vec_p = np.linspace(0, T, num_valuefunctions)
print(t_vec_p)



b = 1 # left end of Domain
horizon = 1
a = -1 # right end of Domain
n = 8 # spacial discretization points that are considered
tau = 1e-3 # time step size
nu = 1 # diffusion constant
lambd = 0.1 # cost parameter
gamma = 0 # discount factor, 0 for no discount
interval_half = 2 # integration area of HJB equation is [-interval_half, interval_half]**n
boundary = 'Neumann' # use 'Neumann' or "Dirichlet
use_full_model = True # if False, model is reduced to r Dimensions
r = n # Model is reduced to r dimensions, only if use_full_model == False
pol_deg = 10
def build_matrices(n, boundary_condition, r = 0):
    s = np.linspace(a, b, n)    # gridpoints
    if boundary_condition == 'Dirichlet':
        print('Dirichlet boundary')
        h = (b-a) / (n+1)
        A = -2*np.diag(np.ones(n), 0) + np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
        A = nu / h**2 * A
        B = np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)
        B = 1/(2*h) * B
        Q = tau*h*np.eye(n)
    elif boundary_condition == 'Neumann':
        print('Neumann boundary')
        h = (b-a)/(n-1)             # step size in space
        A = -2*np.diag(np.ones(n), 0) + np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
        A[0,1] = 2; A[n-1, n-2] = 2
        A = nu / h**2 * A
        Q = tau*h*np.eye(n)
        Q[0,0] /=2; Q[n-1,n-1] /=2  # for neumann boundary
    else:
        print('Wrong boundary!')
    _B = (np.bitwise_and(s > -0.4, s < 0.4))*1.0
    B = np.zeros(shape=(n, 1))
    B[:, 0] = _B
    C = B
    control_dim = B.shape[1]
    R = lambd * np.identity(control_dim)
    P = R*10
    Pi = la.solve_continuous_are(A, B, Q/tau, R)
    return A, B, C, Q, R, P, Pi

def reduce_model(Pi, r, use_full_model, order='lefttoright'):
    n = np.shape(Pi)[0]
    if use_full_model:
        print('did not reduce model, because use_full_model ==', use_full_model)
        proj_full = np.eye(n)
        inj_full = np.eye(n)
        proj = np.eye(n)
        inj = np.eye(n)

    else:
        u, v = la.eigh(Pi)
        u_order = np.argsort(np.abs(u))   # sort by absolute values
        u = np.flip(u[u_order])      # sort from largest to smallest EV
        v = np.flip(v[:,u_order],1)
        if order=='lefttoright':
            inj = v[:, :r]
            inj_full = v
            proj = inj.T
            proj_full = inj_full.T
        else:
            print('This part has to be tested!')
            r_half = int(np.floor(r/2))
            perm = np.zeros(shape=r)
            perm[0] = r_half
            for i0 in range(1, r_half+1):
                if(r_half - i0 >= 0):
                    perm[2*i0 - 1] = r_half - i0
                if(r_half + i0 <r):
                    perm[2*i0] = r_half + i0
            perm_mat = np.zeros((len(perm), len(perm)))
            for idx, i in enumerate(perm):
                perm_mat[int(idx), int(i)] = 1
            inj = np.dot(inj, perm_mat)
            proj = inj.T

            rr = n
            rr_half = int(np.floor(rr/2))
            perm = np.zeros(shape=rr)
            perm[0] = rr_half
            for i0 in range(1, rr_half+1):
                if(rr_half - i0 >= 0):
                    perm[2*i0 - 1] = rr_half - i0
                if(rr_half + i0 <rr):
                    perm[2*i0] = rr_half + i0

            perm_mat = np.zeros((len(perm), len(perm)))
            for idx, i in enumerate(perm):
                perm_mat[int(idx), int(i)] = 1
            #
            inj_full = np.dot(v, perm_mat)
            proj_full = inj_full.T
    return proj, inj, proj_full, inj_full



load = np.zeros([5])
load[0] = lambd; load[1] = gamma; load[2] = interval_half; load[3] = tau; load[4] = n

A, B, C, Q, R, P, Pi = build_matrices(n, boundary, 0)
proj, inj, proj_full, inj_full = reduce_model(Pi, r, use_full_model)
P_discr = P * tau
P_inv = la.inv(P)
R_discr = R * tau
R_inv = la.inv(R)

A_proj = proj @ A @ inj
Pi_proj = proj @ Pi @ inj
Q_proj = proj @ Q @ inj
#


np.save("A_proj", A_proj)
np.save("inj", inj)
np.save("proj", proj)
np.save("inj_full", inj_full)
np.save("proj_full", proj_full)
np.save("A", A)
np.save("data", load)
np.save("B", B)
np.save("C", C)
np.save("Q", Q)
np.save("Pi_proj", Pi_proj)
np.save("Pi_cont", Pi)
np.save("R", R_discr)
np.save("R_inv", R_inv)
np.save("P", P_discr)
np.save("P_inv", P_inv)
np.save('t_vec_p', t_vec_p)
#


