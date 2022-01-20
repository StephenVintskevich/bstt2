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
from misc import  __block, random_homogenous_polynomial_sum_system,random_homogenous_polynomial_sum_system2,zeros_homogenous_polynomial_sum_system,monomial_measures,legendre_measures,Gramian, HkinnerLegendre,random_full_system
from helpers import lennardJonesSamples,lennardJonesSamplesMod, SMat
from als import ALSSystem2
block = __block()

import warnings
warnings.filterwarnings("ignore")

# Parameters
orders = [15]
degree = 6
#maxGroupSize = [1]+[2] +[3]*(order-4)+[2]+[1]
maxGroupSizes = [4,6,8,10]
interactions = [5,7]
trainSampleSizes = [2500,4000,5500,7000]
maxSweeps=8
c = 1.0
mod = 0
res = np.zeros((len(orders),len(trainSampleSizes),len(interactions),len(maxGroupSizes)))
#Model Parameters
exp = 1
for (ii,order) in enumerate(orders):
    for (jj,trainSampleSize) in enumerate(trainSampleSizes):
        for (kk,interaction) in enumerate(interactions):
            for (ll,maxGroupSize) in enumerate(maxGroupSizes):
                print(f'Starting Order" {order} Samples {trainSampleSize} Interaction {interaction} MaxGroupSize {maxGroupSize}')
                sigma = np.ones([order,order])
    
                #Selection Matrix
                S = SMat(interaction,order)
                print(S)
                
                
                coeffs = random_homogenous_polynomial_sum_system2([degree]*order,degree,maxGroupSize,interaction,S)
                print(f"DOFS: {coeffs.dofs()}")
                print(f"Ranks: {coeffs.ranks}")
                print(f"Interaction: {coeffs.interactions}")
                
                
                #train_points,train_values = lennardJonesSamples(order,trainSampleSize,c,sigma,exp)
                train_points,train_values = lennardJonesSamplesMod(order,trainSampleSize,c,exp,mod)
                print(f"Finished drawing samples {trainSampleSize}")
                #train_measures = legendre_measures(train_points, degree,np.float(-c*order),np.float(c*order))
                train_measures = legendre_measures(train_points, degree)
                augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)
                
                
                   
                solver = ALSSystem2(coeffs, augmented_train_measures,  train_values,_verbosity=1)
                solver.maxSweeps = maxSweeps
                solver.targetResidual = 1e-6
                solver.maxGroupSize=maxGroupSize
                solver.run()
                
                testSampleSize = int(2e4)
                test_points,test_values = lennardJonesSamplesMod(order,testSampleSize,c,exp,mod)
                #test_measures =  legendre_measures(test_points, degree,np.float(-c*order),np.float(c*order))
                test_measures =  legendre_measures(test_points, degree)
                augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)
                
                
                values = coeffs.evaluate(augmented_test_measures)
                values2 = coeffs.evaluate(augmented_train_measures)
                print("L2: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)," on training data: ",np.linalg.norm(values2 -  train_values) / np.linalg.norm(train_values))
                res[ii,jj,kk,ll] = np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)
                np.save('exp_4_lennardjones.data',res)