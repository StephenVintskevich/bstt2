#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:25:27 2021

@author: goette
"""

import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


import numpy as np 
from misc import  __block,legendre_measures,random_homogenous_polynomial_sum_system2
from helpers import fermi_pasta_ulam,SMat
from als import ALSSystem2
block = __block()

import warnings
warnings.filterwarnings("ignore")

order = 50  
degree = 3
maxGroupSize = 2
#interaction = [3]+ [4] + [5]*(order-4) + [4] + [3]
interaction = 5
#interaction = [2]*order

trainSampleSize = 5000
maxSweeps=20
ranks = [4]*(order-1)

S = SMat(interaction,order)



train_points,train_values = fermi_pasta_ulam(order,trainSampleSize)
print(train_points.shape)
print(train_values.shape)
train_measures = legendre_measures(train_points, degree)
print(train_measures.shape)
augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)

bstt = random_homogenous_polynomial_sum_system2([degree]*order,degree,maxGroupSize,interaction,S)
print(f"DOFS: {bstt.dofs()}")
print(f"Ranks: {bstt.ranks}")
print(f"Interaction: {bstt.interactions}")


   
solver = ALSSystem2(bstt, augmented_train_measures,  train_values,_verbosity=1)
solver.maxSweeps = maxSweeps
solver.targetResidual = 1e-6
solver.maxGroupSize=maxGroupSize
solver.run()

testSampleSize = int(2e4)
test_points,test_values =fermi_pasta_ulam(order,testSampleSize)
test_measures = legendre_measures(test_points, degree)
augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)


values = bstt.evaluate(augmented_test_measures)
values2 = bstt.evaluate(augmented_train_measures)
print("L2: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)," on training data: ",np.linalg.norm(values2 -  train_values) / np.linalg.norm(train_values))




