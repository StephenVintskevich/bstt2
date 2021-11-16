#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 09:55:05 2021

@author: goette
"""

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from tilde_r import calc_total_reward,calc_tilde_r

import optimize
from misc import random_homogenous_polynomial_sum,  legendre_measures, Gramian, HkinnerLegendre  #, hermite_measures
from als import ALS
import numpy as np
import copy
 

data = np.load('data.npy')
t_vec = np.load("t_vec_p.npy")

order = int(data[4])
degree = 7
maxGroupSize = 2
maxSweeps = 8
tol =1e-4
maxPolIt = 3

print(f"Order {order}")
print(f"degree {degree}")

#generate sample data
trainSampleSize = int(1000)
print(f"Sample Size {trainSampleSize}")
train_points = 2*np.random.rand(trainSampleSize,order)-1
train_measures = legendre_measures(train_points, degree)
augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)

testSampleSize = int(100)
test_points = 2*np.random.rand(testSampleSize,order)-1
test_measures = legendre_measures(test_points, degree)
augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)


f = lambda xs: np.linalg.norm(xs, axis=1)**2
end_values = f(train_points)


#calculate local gramians
localH1Gramians = [Gramian(degree+1,HkinnerLegendre(1)) for i in range(order)]
localH1Gramians.append(np.ones([degree+1,degree+1]))
localL2Gramians = [np.eye(degree+1) for i in range(order)]
localL2Gramians.append(np.ones([degree+1,degree+1]))

vlist = []
#set end condititon


for t in np.flipud(t_vec):
    print(f"Time point: {t}")
    if t == t_vec[-1]:
        vend = random_homogenous_polynomial_sum([degree]*order, degree, maxGroupSize)
        vend.assume_corePosition(vend.order-1)
        while vend.corePosition > 0:
            vend.move_core('left')
        vend.components[0] = np.zeros(vend.components[0].shape)
        print(f"DOFS {vend.dofs()}")
        solver = ALS(vend, augmented_train_measures,  end_values,_localL2Gramians=localL2Gramians,_localH1Gramians=localH1Gramians,_verbosity=1)
        solver.maxSweeps = 10
        solver.targetResidual = 1e-8
        solver.run()
        vlist.append(vend)
    else:
        vnext = copy.deepcopy(vlist[-1])
        vlist.append(vnext)
        
        
        count = 0
        while count < maxPolIt:
            #calculate rhs
            print(f"Start Calculating RHS")
            rhs = calc_tilde_r(t,train_points.T,vlist)
            print(f"Finished Calculating RHS")
            values = vlist[-1].evaluate(augmented_train_measures)
            print(f"DOFS { vlist[-1].dofs()}")

            err = np.linalg.norm(values - rhs)/float(trainSampleSize)
            print(f"Error {err}")
            if err < tol:
                break
            #update value function
            solver = ALS(vlist[-1], augmented_train_measures,  rhs,_localL2Gramians=localL2Gramians,_localH1Gramians=localH1Gramians,_verbosity=1)
            solver.maxSweeps = maxSweeps
            solver.targetResidual = 1e-5
            solver.run()
            count+=1



test_values = calc_total_reward(test_points.T,vlist)
for i in range(testSampleSize):
    print(f"Test Value {i}: {test_values[i]}")
