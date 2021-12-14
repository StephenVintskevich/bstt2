#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:25:27 2021

@author: goette
"""
import numpy as np 
from misc import  __block, random_homogenous_polynomial_sum_system,random_homogenous_polynomial_sum,zeros_homogenous_polynomial_sum_system,monomial_measures,legendre_measures,Gramian, HkinnerLegendre,random_full_system
from helpers import magneticDipolesSamples, selectionMatrix4,selectionMatrix3,selectionMatrix3dense
from als_l1_test import ALSSystem
block = __block()

import warnings
warnings.filterwarnings("ignore")

order = 12
degree = 9
interaction = [3]+ [4]*(order-2)  + [3]
trainSampleSize = 30000
maxSweeps=10
ranks = [6]*(order-1)


M = np.ones(order)
I = np.ones(order)
x = np.linspace(0,2*(order-1),order)




train_points,train_values = magneticDipolesSamples(order,trainSampleSize,M,x,I)
train_points = train_points.T
train_values = train_values.T
print(train_points.shape)
print(train_values.shape)
train_measures = legendre_measures(train_points, degree,-np.pi,np.pi)
print(train_measures.shape)

bstt = random_full_system([degree]*order,interaction,ranks,selectionMatrix3dense)
print(f"DOFS: {bstt.dofs()}")
print(f"Ranks: {bstt.ranks}")
print(f"Interaction: {bstt.interaction}")
localH1Gramians = [Gramian(degree+1,HkinnerLegendre(1)) for i in range(order)]
localH1Gramians.append(np.ones([degree+1,degree+1]))
localL2Gramians = [np.eye(degree+1) for i in range(order)]
localL2Gramians.append(np.ones([degree+1,degree+1]))

   
solver = ALSSystem(bstt, train_measures,  train_values,_verbosity=1)
solver.maxSweeps = maxSweeps
solver.targetResidual = 1e-6
solver.run()

testSampleSize = int(2e4)
test_points,test_values = magneticDipolesSamples(order,testSampleSize,M,x,I)
test_points = test_points.T
test_values = test_values.T
test_measures = legendre_measures(test_points, degree,-np.pi,np.pi)


values = bstt.evaluate(test_measures)
values2 = bstt.evaluate(train_measures)
print("L2: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)," on training data: ",np.linalg.norm(values2 -  train_values) / np.linalg.norm(train_values))
