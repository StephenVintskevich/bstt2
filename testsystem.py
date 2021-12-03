#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:25:27 2021

@author: goette
"""
import numpy as np 
from misc import  __block, random_homogenous_polynomial_sum_system,random_homogenous_polynomial_sum,zeros_homogenous_polynomial_sum_system,monomial_measures,legendre_measures,Gramian, HkinnerLegendre,random_full_system
from helpers import fermi_pasta_ulam
from als_l1_test import ALSSystem
from bstt import Block, BlockSparseTT
block = __block()

#import warnings
#warnings.filterwarnings("ignore")

order = 8
degree = 3
maxGroupSize = 2
interaction = [3]+ [4] + [5]*(order-4) + [4] + [3]
trainSampleSize = 2000
maxSweeps=20
eq = 3

def selectionMatrix(k,_numberOfEquations):
    assert k >= 0 and k < _numberOfEquations+1
    if k == 0:
        Smat = np.zeros([3,_numberOfEquations])
        Smat[2,0] = 1
        Smat[1,1] = 1
        for i in range(2,_numberOfEquations):
            Smat[0,i] = 1
    elif k == 1:
        Smat = np.zeros([4,_numberOfEquations])
        Smat[3,0] = 1
        Smat[2,1] = 1
        Smat[1,2] = 1
        for i in range(3,_numberOfEquations):
            Smat[0,i] = 1
    elif k == _numberOfEquations - 2:
        Smat = np.zeros([4,_numberOfEquations])
        for i in range(0,k-1):
            Smat[3,i] = 1
        Smat[2,k-1] = 1
        Smat[1,k] = 1
        Smat[0,k+1] = 1
    elif k == _numberOfEquations - 1:
        Smat = np.zeros([3,_numberOfEquations])
        for i in range(0,k-1):
            Smat[2,i] = 1
        Smat[1,k-1] = 1
        Smat[0,k] = 1
    elif k == _numberOfEquations:
         Smat = np.zeros([1,_numberOfEquations])
         for i in range(0,_numberOfEquations):
             Smat[0,i] = 1
    else:        
        Smat = np.zeros([5,_numberOfEquations])
        for i in range(0,k-1):
            Smat[4,i] = 1
        Smat[3,k-1] = 1
        Smat[2,k] = 1
        Smat[1,k+1] = 1
        for i in range(k+2,_numberOfEquations):
            Smat[0,i] = 1
    return Smat


train_points,train_values = fermi_pasta_ulam(order,trainSampleSize)
train_points = train_points.T
train_values = train_values.T
print(train_points.shape)
print(train_values.shape)
train_measures = legendre_measures(train_points, degree)
print(train_measures.shape)
augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)

bstt = random_homogenous_polynomial_sum_system([degree]*order,interaction,degree,maxGroupSize,selectionMatrix)

print(f"DOFS: {bstt.dofs()}")
print(f"Ranks: {bstt.ranks}")
print(f"Interaction: {bstt.interaction}")
localH1Gramians = [Gramian(degree+1,HkinnerLegendre(1)) for i in range(order)]
localH1Gramians.append(np.ones([degree+1,degree+1]))
localL2Gramians = [np.eye(degree+1) for i in range(order)]
localL2Gramians.append(np.ones([degree+1,degree+1]))

   
solver = ALSSystem(bstt, augmented_train_measures,  train_values,_localL2Gramians=localL2Gramians,_localH1Gramians=localH1Gramians,_verbosity=2)
solver.maxSweeps = maxSweeps
solver.targetResidual = 1e-6
#solver.increaseRanks=increaseRanks
solver.maxGroupSize=maxGroupSize
solver.run()

testSampleSize = int(2e4)
test_points,test_values =fermi_pasta_ulam(order,testSampleSize)
test_points = test_points.T
test_values = test_values.T
test_measures = legendre_measures(test_points, degree)
augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)


values = bstt.evaluate(augmented_test_measures)
values2 = bstt.evaluate(augmented_train_measures)
print("L2: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)," on training data: ",np.linalg.norm(values2 -  train_values) / np.linalg.norm(train_values))
