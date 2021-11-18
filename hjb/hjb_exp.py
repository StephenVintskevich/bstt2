#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 09:55:05 2021

@author: goette
"""

# , hermite_measures
from misc import random_homogenous_polynomial_sum,  legendre_measures, Gramian, HkinnerLegendre
import ode
import copy
import numpy as np
from als import ALS
import optimize
from tilde_r import calc_tilde_r, calc_total_reward
import os
import sys
import set_dynamics

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


data = np.load('data.npy')
t_vec = np.load("t_vec_p.npy")
T = t_vec[-1]
load_me = np.load('data.npy')
tau = load_me[3]#t_vec[1]-t_vec[0]
print(tau)
order = int(data[4])
degree = 3
maxGroupSize = 2
maxSweeps = 2
tol = 1e-4
maxPolIt = 1
Schloegel_ode = ode.Ode()
print(f"Order {order}")
print(f"degree {degree}")

# generate sample data
trainSampleSize = int(100)#int(3000)
print(f"Sample Size {trainSampleSize}")
train_points = 2*np.random.rand(trainSampleSize, order)-1
train_measures = legendre_measures(train_points, degree)
augmented_train_measures = np.concatenate(
    [train_measures, np.ones((1, trainSampleSize, degree+1))], axis=0)


def f(xs): return Schloegel_ode.calc_end_reward(0, xs.T)#np.linalg.norm(xs, axis=1)**2


end_values = f(train_points)


# calculate local gramians
localH1Gramians = [Gramian(degree+1, HkinnerLegendre(1)) for i in range(order)]
localH1Gramians.append(np.ones([degree+1, degree+1]))
localL2Gramians = [np.eye(degree+1) for i in range(order)]
localL2Gramians.append(np.ones([degree+1, degree+1]))

vlist = []
# set end condititon


for t in np.flipud(t_vec):
    print(f"Time point: {t}")
    if t == t_vec[-1]:
        vend = random_homogenous_polynomial_sum(
            [degree]*order, degree, maxGroupSize)
        vend.assume_corePosition(vend.order-1)
        while vend.corePosition > 0:
            vend.move_core('left')
        vend.components[0] = np.zeros(vend.components[0].shape)
        print(f"DOFS {vend.dofs()}")
        solver = ALS(vend, augmented_train_measures,  end_values,
                     _localL2Gramians=localL2Gramians, _localH1Gramians=localH1Gramians, _verbosity=1)
        solver.maxSweeps = 10
        solver.targetResidual = 1e-8
        solver.run()
        vlist.append(vend)
    else:
        vnext = copy.deepcopy(vlist[-1])
        vlist.append(vnext)

        count = 0
        while count < maxPolIt:
            # calculate rhs
            print(f"Start Calculating RHS")
            rhs = calc_tilde_r(t, train_points.T, vlist)
            print(f"Finished Calculating RHS")
            values = vlist[-1].evaluate(augmented_train_measures)
            print(f"DOFS { vlist[-1].dofs()}")

            err = np.linalg.norm(values - rhs)/float(trainSampleSize)
            print(f"Error {err}")
            if err < tol:
                break
            # update value function
            solver = ALS(vlist[-1], augmented_train_measures,  rhs,
                         _localL2Gramians=localL2Gramians, _localH1Gramians=localH1Gramians, _verbosity=1)
            solver.maxSweeps = maxSweeps
            solver.targetResidual = 1e-5
            solver.run()
            count += 1

print("vlist", len(vlist))
# evaluating the result of policy iteration
x = 2*np.random.rand(1,order)-1


#open loop solver from Leon
step_size = .05
step_size_before = 0.002
max_iter = 1e5
grad_tol = 1e-8
steps = np.linspace(0, T, int(T/tau)+1)
print('steps',len(steps),'tau',tau)
m = len(steps)
rew_hjb, u_hjb = calc_total_reward(x.T, steps,1*vlist)
print('u_hjb',u_hjb.shape)
control_dim = 1
optimize_params = [step_size, step_size_before, max_iter, grad_tol]
opt_u_item = optimize.Open_loop_solver(Schloegel_ode, optimize_params)
opt_u_item.initialize_new(Schloegel_ode.calc_end_reward_grad, steps)

def calc_opt(x0, u0, calc_cost):
    
    x_vec, u_vec = opt_u_item.calc_optimal_control(x0, u0)
    cost = 1/2*calc_cost(0, x_vec[:, 0], u_vec[:, 0])
    for i0 in range(len(steps)-1):
        add_cost = calc_cost(steps[i0+1], x_vec[:,i0+1 ], u_vec[:, i0+1])
        cost += add_cost
    cost -= add_cost/2
    cost += Schloegel_ode.calc_end_reward(0, x_vec[:,-1])
    return x_vec.T, u_vec.T, cost
x_opt, u_opt, cost_opt = calc_opt(x.T, u_hjb, Schloegel_ode.calc_reward)
print("cost hjb", rew_hjb, 'cost opt', cost_opt)