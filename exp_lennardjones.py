#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:25:27 2021

@author: goette
"""
import numpy as np 
from misc import  __block, random_homogenous_polynomial_sum_system,random_homogenous_polynomial_sum_system2,zeros_homogenous_polynomial_sum_system,monomial_measures,legendre_measures,Gramian, HkinnerLegendre,random_full_system
from helpers import lennardJonesSamples, SMat
from als_l1_test import ALSSystem2
block = __block()

import warnings
warnings.filterwarnings("ignore")

# Parameters
order = 10
degree = 6
#maxGroupSize = [1]+[2] +[3]*(order-4)+[2]+[1]
maxGroupSize = 6
interaction = 5
trainSampleSize = 20000
maxSweeps=10
c = 1

#Model Parameters
exp = 2
sigma = np.ones([order,order])

#Selection Matrix
S = SMat(interaction,order)
print(S)


coeffs = random_homogenous_polynomial_sum_system2([degree]*order,degree,maxGroupSize,interaction,S)
print(f"DOFS: {coeffs.dofs()}")
print(f"Ranks: {coeffs.ranks}")
print(f"Interaction: {coeffs.interactions}")


train_points,train_values = lennardJonesSamples(order,trainSampleSize,c,sigma,exp)
print("Finished drawing samples")
train_measures = legendre_measures(train_points, degree,-c*order,c*order)
augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)


   
solver = ALSSystem2(coeffs, augmented_train_measures,  train_values,_verbosity=1)
solver.maxSweeps = maxSweeps
solver.targetResidual = 1e-6
solver.maxGroupSize=maxGroupSize
solver.run()

testSampleSize = int(2e4)
test_points,test_values = lennardJonesSamples(order,testSampleSize,c,sigma,exp)
test_measures =  legendre_measures(test_points, degree,-c*order,c*order)
augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)


values = coeffs.evaluate(augmented_test_measures)
values2 = coeffs.evaluate(augmented_train_measures)
print("L2: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)," on training data: ",np.linalg.norm(values2 -  train_values) / np.linalg.norm(train_values))
