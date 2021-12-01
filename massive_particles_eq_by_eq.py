#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:25:27 2021

@author: goette
"""
import numpy as np 
from misc import  __block, random_homogenous_polynomial_sum_system,random_homogenous_polynomial_sum,zeros_homogenous_polynomial_sum_system,monomial_measures,legendre_measures,Gramian, HkinnerLegendre,random_full_system
from helpers import massive_particles
from als import ALSSystem,ALS
from bstt import Block, BlockSparseTT
block = __block()

import warnings
warnings.filterwarnings("ignore")

order = 5
degree = 6
maxGroupSize = 5
trainSampleSize = 2000
maxSweeps=20
G = 1
m = [0.0001]*order
r  = 10


train_points,train_values = massive_particles(order,trainSampleSize,G,m,r)
train_points = train_points.T
train_values = train_values.T
print(train_points.shape)
print(train_values.shape)
train_measures = legendre_measures(train_points, degree)
print(train_measures.shape)
augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)



localH1Gramians = [Gramian(degree+1,HkinnerLegendre(1)) for i in range(order)]
localH1Gramians.append(np.ones([degree+1,degree+1]))
localL2Gramians = [np.eye(degree+1) for i in range(order)]
localL2Gramians.append(np.ones([degree+1,degree+1]))
for k in range(order):
    bstt = random_homogenous_polynomial_sum([degree]*order,degree,maxGroupSize)
    print(f"DOFS: {bstt.dofs()}")

    #solver = ALSSystem(bstt, train_measures,  train_values,_localL2Gramians=localL2Gramians,_localH1Gramians=localH1Gramians,_verbosity=1)
    solver = ALS(bstt, augmented_train_measures,  train_values[:,k],_verbosity=1)
    solver.maxSweeps = maxSweeps
    solver.targetResidual = 1e-3
    #solver.increaseRanks=increaseRanks
    solver.maxGroupSize=maxGroupSize
    solver.run()

    testSampleSize = int(1e5)
    test_points,test_values =massive_particles(order,testSampleSize,G,m,r)
    test_points = test_points.T
    test_values = test_values.T
    test_measures = legendre_measures(test_points, degree)
    augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)
    
    
    values = bstt.evaluate(augmented_test_measures)
    values2 = bstt.evaluate(augmented_train_measures)
    print("L1 sparse: ",np.linalg.norm(values -  test_values[:,k]) / np.linalg.norm(test_values[:,k])," on training data: ",np.linalg.norm(values2 -  train_values[:,k]) / np.linalg.norm(train_values[:,k]))
