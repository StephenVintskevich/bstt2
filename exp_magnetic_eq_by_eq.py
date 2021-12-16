#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:25:27 2021

@author: goette
"""
import numpy as np 
from misc import  __block,sinecosine_measures, random_fixed_variable_sum,random_homogenous_polynomial_sum,zeros_homogenous_polynomial_sum_system,monomial_measures,legendre_measures,Gramian, HkinnerLegendre,random_full_system
from helpers import magneticDipolesSamples
from als import ALSSystem,ALS
from als_old import ALS as ALSOLD
from bstt import Block, BlockSparseTT
import copy
block = __block()

import warnings
warnings.filterwarnings("ignore")

order = 20
degree = 2
maxGroupSize = 2
trainSampleSize = 10000
maxSweeps=20
M = np.ones(order)
I = np.ones(order)
x = np.linspace(0,1*(order-1),order)

train_points,train_values = magneticDipolesSamples(order,trainSampleSize,M,x,I)
train_points = train_points.T
train_values = train_values.T
train_measures = sinecosine_measures(train_points)
augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)


localH1Gramians = [Gramian(degree+1,HkinnerLegendre(1)) for i in range(order)]
localH1Gramians.append(np.ones([degree+1,degree+1]))
localL2Gramians = [np.eye(degree+1) for i in range(order)]
localL2Gramians.append(np.ones([degree+1,degree+1]))
for k in range(1):
    bstt = random_fixed_variable_sum([degree]*order,degree,maxGroupSize)
    bstt2 = copy.deepcopy(bstt)
    print(f"DOFS: {bstt.dofs()}")
    print(f"Ranks: {bstt.ranks}")
    #solver = ALS(bstt, augmented_train_measures,  train_values[:,k], _localL2Gramians=localL2Gramians, _localH1Gramians=localH1Gramians,_verbosity=1)
    solver = ALSOLD(bstt, augmented_train_measures,  train_values[:,k],_verbosity=1)
    solver.maxSweeps = maxSweeps
    solver.targetResidual = 2e-8
    #solver.increaseRanks=increaseRanks
    solver.maxGroupSize=maxGroupSize
    solver.run()

    testSampleSize = int(1e4)
    test_points,test_values = magneticDipolesSamples(order,testSampleSize,M,x,I)
    test_points = test_points.T
    test_values = test_values.T
    test_measures = sinecosine_measures(train_points)
    augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)
    
    
    values = bstt.evaluate(augmented_test_measures)
    values2 = bstt.evaluate(augmented_train_measures)
    print("L1 sparse: ",np.linalg.norm(values -  test_values[:,k]) / np.linalg.norm(test_values[:,k])," on training data: ",np.linalg.norm(values2 -  train_values[:,k]) / np.linalg.norm(train_values[:,k]))


