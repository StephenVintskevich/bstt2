#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 09:55:05 2021

@author: goette
"""

# , hermite_measures
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from misc import random_homogenous_polynomial_sum,  legendre_measures, Gramian, HkinnerLegendre
import ode
import copy
import numpy as np
from als import ALS
import optimize
from tilde_r import calc_tilde_r, calc_total_reward
import set_dynamics




data = np.load('data.npy')
t_vec = np.load("t_vec_p.npy")
T = t_vec[-1]
load_me = np.load('data.npy')
tau = load_me[3]#t_vec[1]-t_vec[0]
print(tau)
order = int(data[4])
degree = 8
maxGroupSize = 5
maxSweeps = 20
tol = 1e-4
maxPolIt = 10
Schloegel_ode = ode.Ode()
print(f"Order {order}")
print(f"degree {degree}")
a=-2
b=2

#generate sample data
N = 4000
trainSampleSize = int(N)
print(f"Sample Size {trainSampleSize}")
train_points = (b-a)*np.random.rand(trainSampleSize, order)+a
train_measures = legendre_measures(train_points, degree, _a=a,_b=b)
augmented_train_measures = np.concatenate(
    [train_measures, np.ones((1, trainSampleSize, degree+1))], axis=0)


def f(xs): return Schloegel_ode.calc_end_reward(0, xs.T)#np.linalg.norm(xs, axis=1)**2





testSampleSize = int(100)
test_points = 2*np.random.rand(testSampleSize,order)-1
test_measures = legendre_measures(test_points, degree,_a=a,_b=b)
augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)


#f = lambda xs: np.linalg.norm(xs, axis=1)**2
end_values = f(train_points)


# calculate local gramians
localH1Gramians = [Gramian(degree+1, HkinnerLegendre(1),_a=a,_b=b) for i in range(order)]
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
            # update v  alue function
            solver = ALS(vlist[-1], augmented_train_measures,  rhs,
                         _localL2Gramians=localL2Gramians, _localH1Gramians=localH1Gramians, _verbosity=1)
            solver.maxSweeps = maxSweeps
            solver.targetResidual = 1e-5
            solver.run()
            count += 1

print("vlist", len(vlist))
# evaluating the result of policy iteration
# evaluating the result of policy iteration
x = 2*np.random.rand(1,order)-1


#open loop solver from Leon
step_size = .5
step_size_before = 0.02
max_iter = 1e5
grad_tol = 1e-15
steps = np.linspace(0, T, int(T/tau)+1)
print('steps',len(steps),'tau',tau)
m = len(steps)
rew_hjb, u_hjb, x_hjb = calc_total_reward(x.T, steps,1*vlist)
#print(last_rew, Schloegel_ode.calc_end_reward(0, x_hjb[:,-1]),x_end,x_hjb[:,-1],x_wub)
print('u_hjb',u_hjb.shape)
control_dim = 1
optimize_params = [step_size, step_size_before, max_iter, grad_tol]
opt_u_item = optimize.Open_loop_solver(Schloegel_ode, optimize_params)
opt_u_item.initialize_new(Schloegel_ode.calc_end_reward_grad, steps)

def calc_opt(x0, u0, calc_cost, x_opt):
    print('shapes', x0.shape,u0.shape, x_opt.shape)
    x_vec, u_vec = opt_u_item.calc_optimal_control(x0, u0)
    print('shapes2', x_vec.shape, u_vec.shape)
    u_hjb = u0
    cost = 1/2*calc_cost(0, x_vec[:, 0], u_vec[:, 0])
    cost1 = 1/2*calc_cost(0, x_vec[:, 0], u_hjb[:, 0])
    cost2 = 1/2*calc_cost(0, x_opt[:, 0], u_hjb[:, 0])

    for i0 in range(len(steps)-1):
        add_cost = calc_cost(steps[i0+1], x_vec[:,i0+1 ], u_vec[:, i0+1])
        add_cost1 = calc_cost(steps[i0+1], x_vec[:,i0+1 ], u_hjb[:, i0+1])
        add_cost2 = calc_cost(steps[i0+1], x_opt[:,i0+1 ], u_hjb[:, i0+1])
        cost += add_cost
        cost1 += add_cost1
        cost2 += add_cost2

    cost -= add_cost/2
    cost += Schloegel_ode.calc_end_reward(0, x_vec[:,-1])
    cost1 -= add_cost1/2
    cost1 += Schloegel_ode.calc_end_reward(0, x_vec[:,-1])
    cost2 -= add_cost2/2
    cost2 += Schloegel_ode.calc_end_reward(0, x_opt[:,-1])
    return x_vec.T, u_vec.T, cost, cost1, cost2, Schloegel_ode.calc_end_reward(0, x_opt[:,-1])
x_opt, u_opt, cost_opt,cost1 ,cost2, last_rew_2= calc_opt(x.T, u_hjb+0.0001*np.random.rand(u_hjb.shape[0],u_hjb.shape[1]), Schloegel_ode.calc_reward,x_hjb)
print("cost hjb", rew_hjb, 'cost opt', cost_opt, 'cost1',cost1, 'cost2', cost2,'norm(x_opt-x_hjb)', np.linalg.norm(x_opt.T-x_hjb)/np.linalg.norm(x_opt))#,'last_rew 1 2', last_rew,last_rew_2, 'norm(x_opt-x_hjb)', np.linalg.norm(x_opt.T-x_hjb)/np.linalg.norm(x_opt))
      
#+0.01*np.random.rand(u_hjb.shape[0],u_hjb.shape[1])
   



#test_values = calc_total_reward(test_points.T,vlist)

