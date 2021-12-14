#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:25:27 2021

@author: goette
"""
import numpy as np 
from misc import  __block, random_homogenous_polynomial_sum_system,random_homogenous_polynomial_sum,zeros_homogenous_polynomial_sum_system,monomial_measures,legendre_measures,Gramian, HkinnerLegendre,random_full_system
from helpers import lennardJonesSamples, selectionMatrix4,selectionMatrix3
from als_l1_test import ALSSystem
block = __block()

import warnings
warnings.filterwarnings("ignore")

order = 8
degree = 14
maxGroupSize = 2
interaction = [4]+ [5] + [6]+[7]*(order-6) +[6]+[5] + [4]
#interaction = [3]+ [4] + [5]*(order-4) +[4] + [3]
trainSampleSize = 20000
maxSweeps=5
ranks = [4]*(order-1)
c = 1
sigma = np.ones([order,order])


bstt = random_homogenous_polynomial_sum_system([degree]*order,interaction,degree,maxGroupSize,selectionMatrix4)
print(f"DOFS: {bstt.dofs()}")
print(f"Ranks: {bstt.ranks}")
print(f"Interaction: {bstt.interaction}")


train_points,train_values = lennardJonesSamples(order,trainSampleSize,c,sigma)
train_points = train_points.T
train_values = train_values.T
print(train_points.shape)
print(train_values.shape)
train_measures = legendre_measures(train_points, degree,-c*order,c*order)
print(train_measures.shape)
augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)

localH1Gramians = [Gramian(degree+1,HkinnerLegendre(1)) for i in range(order)]
localH1Gramians.append(np.ones([degree+1,degree+1]))
localL2Gramians = [np.eye(degree+1) for i in range(order)]
localL2Gramians.append(np.ones([degree+1,degree+1]))

   
solver = ALSSystem(bstt, augmented_train_measures,  train_values,_verbosity=1)
solver.maxSweeps = maxSweeps
solver.targetResidual = 1e-6
solver.maxGroupSize=maxGroupSize
solver.run()

testSampleSize = int(2e4)
test_points,test_values = lennardJonesSamples(order,testSampleSize,c,sigma)
test_points = test_points.T
test_values = test_values.T
test_measures =  legendre_measures(test_points, degree,-c*order,c*order)
augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)


values = bstt.evaluate(augmented_test_measures)
values2 = bstt.evaluate(augmented_train_measures)
print("L2: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)," on training data: ",np.linalg.norm(values2 -  train_values) / np.linalg.norm(train_values))
