#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:25:27 2021

@author: goette
"""
import numpy as np 
from misc import  __block, sinecosine_measures, random_homogenous_polynomial_sum_system2,random_fixed_variable_sum_system2,legendre_measures
from helpers import magneticDipolesSamples, SMat
from als_l1_test import ALSSystem2
from bstt import BlockSparseTT
block = __block()

import warnings
warnings.filterwarnings("ignore")

# Parameters
order = 40
degree = 2
maxGroupSize = [1]+[2] +[3]*(order-4)+[2]+[1]
interaction = 9
trainSampleSize = 10000
maxSweeps=25
b = np.pi
# Model Parameters
M = np.ones(order)
I = np.ones(order)
x = np.linspace(0,1*(order-1),order)

S = SMat(interaction,order)
print(S)

# Training Data Generation
train_points,train_values = magneticDipolesSamples(order,trainSampleSize,M,x,I)
#train_measures = legendre_measures(train_points, degree,-b,b)
train_measures = sinecosine_measures(train_points)
augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)

# Model initialization (bsTT)
#coeffs = random_homogenous_polynomial_sum_system2([degree]*order,degree,maxGroupSize,interaction,S)
coeffs = random_fixed_variable_sum_system2([degree]*order,degree,maxGroupSize,interaction,S)
print(f"DOFS: {coeffs.dofs()}")
print(f"Ranks: {coeffs.ranks}")
print(f"Interaction: {coeffs.interactions}")

# Solving
solver = ALSSystem2(coeffs, augmented_train_measures,  train_values,_verbosity=1)
solver.maxSweeps = maxSweeps
solver.targetResidual = 1e-4
solver.maxGroupSize=maxGroupSize
solver.run()

# Testing Data Generation
testSampleSize = int(2e4)
test_points,test_values = magneticDipolesSamples(order,testSampleSize,M,x,I)
#test_measures = legendre_measures(test_points, degree,-b,b)
test_measures = sinecosine_measures(test_points)
augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)

# Error Evaluation
values = coeffs.evaluate(augmented_test_measures)
values2 = coeffs.evaluate(augmented_train_measures)
print("l2 on test data: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)," on training data: ",np.linalg.norm(values2 -  train_values) / np.linalg.norm(train_values))
for k in range(order):
    print("l2 on test data: ",np.linalg.norm(values[:,k]-  test_values[:,k]) / np.linalg.norm(test_values[:,k])," on training data: ",np.linalg.norm(values2[:,k] -  train_values[:,k]) / np.linalg.norm(train_values[:,k]))