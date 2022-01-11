#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 12:34:27 2022

@author: goette
"""

import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


from misc import random_homogenous_polynomial_sum_grad,random_homogenous_polynomial_sum,legendre_measures_grad2
import numpy as np
from als import ALSGrad

#Parameters
order = 8
degree = 2
maxGroupSize = 2
trainSampleSize=1000
maxSweeps=5

# Dynamics
bandSize = 3
decay = 5
A = np.zeros([order,order])
A += np.diag(np.random.rand(order))
eps = 1
for i in range(1,bandSize):
    eps/=decay
    A += eps*np.diag(np.random.rand(order-i),-i)
    A += eps*np.diag(np.random.rand(order-i),i)
print(A)

train_points = 2*np.random.rand(trainSampleSize,order)-1
train_values = train_points@A.T
print(train_points.shape)
print(train_values.shape)

train_measures,train_measures_grad = legendre_measures_grad2(train_points, degree)
print( train_measures.shape)
print( train_measures_grad.shape)
augmented_train_measures =  \
    np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0) 
    
print(augmented_train_measures.shape)


coeffs = random_homogenous_polynomial_sum_grad([degree]*order,degree,maxGroupSize)
print(f"DOFS: {coeffs.dofs()}")
print(f"Ranks: {coeffs.ranks}")
measure = np.zeros((order+1,1,degree+1))
measure[:,:,0] = np.ones((order+1,1))
measure[order,:,:] = np.ones((1,degree+1))
print(coeffs.evaluate(measure))
#measure[3,:,0] = 0
#measure[3,:,1] = 1
measure[4,:,0] = 0
measure[4,:,1] = 1
print(coeffs.evaluate(measure))


   
solver = ALSGrad(coeffs, augmented_train_measures,train_measures_grad,  train_values,_verbosity=1)
solver.maxSweeps = maxSweeps
solver.targetResidual = 1e-4
#solver.increaseRanks=increaseRanks
solver.maxGroupSize=maxGroupSize
solver.run()

testSampleSize = int(2e4)
test_points = 2*np.random.rand(trainSampleSize,order)-1
test_values = train_points@A.T
test_measures,test_measures_grad = legendre_measures_grad2(test_points, degree)
augmented_test_measures =  \
    np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0) 


def res(bstt, measure, measure_grad,values):
    res = 0
    for pos in range(bstt.order-1):
        tmp_measures = measure
        tmp_measures[pos] =measure_grad[pos]
        tmp_res = bstt.evaluate(tmp_measures)
        res += np.linalg.norm(tmp_res -  values[pos])**2    
    return np.sqrt(res) / np.linalg.norm(values)



print("L2: ",res(coeffs,test_measures,test_measures_grad,test_values)," on training data: ",res(coeffs,train_measures,train_measures_grad,train_values))

