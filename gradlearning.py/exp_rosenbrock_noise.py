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


from misc import random_homogenous_polynomial_sum_grad,random_homogenous_polynomial_sum,legendre_measures_grad2,monomial_measures_grad2,legendre_measures
import numpy as np
from als import ALSGrad,ALS

#Parameters
order = 12
degree = 4
maxGroupSize = 3
trainSampleSize=2000
maxSweeps=15
sigma = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]

# Dynamics
def f(x):
    N = x.shape[1]
    res = 0
    for i in range(N-1):
        res += 100*(x[:,i+1]-x[:,i]**2)**2+(1-x[:,i])**2
    return res

def df(x):
    N = x.shape[1]
    res = [0]*N
    for i in range(N-1):
        res[i]+=(-400*(x[:,i+1]-x[:,i]**2)*x[:,i]-2*(1-x[:,i]))
    for i in range(1,N):
        res[i]+= 200*(x[:,i]-x[:,i-1]**2)
    return np.array(res).T
result = []
for s in sigma:
    res_tmp = []
    for t in sigma:
        print(f"order {order} Groupsize {maxGroupSize} Samplesize {trainSampleSize} sigma input {s} sigma output {t}")
        # exact values
        train_points = 2*np.random.rand(trainSampleSize,order)-1       
        train_values = df(train_points) 
        train_values2 = f(train_points)   
        # added noise
        train_points += np.random.normal(0,s,train_points.shape)
        train_values += np.random.normal(0,t,train_values.shape)
        train_values2 += np.random.normal(0,t,train_values2.shape)
        
        #calculate measures
        train_measures,train_measures_grad = legendre_measures_grad2(train_points, degree)
        #train_measures,train_measures_grad = monomial_measures_grad2(train_points, degree)
        augmented_train_measures =  \
            np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0) 
        augmented_train_measures_grad =  \
            np.concatenate([train_measures_grad, np.ones((1,trainSampleSize,degree+1))], axis=0)
        
        
        coeffs = random_homogenous_polynomial_sum_grad([degree]*order,degree,maxGroupSize)
        coeffs2 = random_homogenous_polynomial_sum([degree]*order,degree,maxGroupSize)
        print(f"DOFS: {coeffs.dofs()}")
        print(f"Ranks: {coeffs.ranks}")
        
        
        solver = ALSGrad(coeffs, augmented_train_measures,augmented_train_measures_grad,  train_values,_verbosity=1)
        solver.maxSweeps = maxSweeps
        solver.targetResidual = 1e-6
        solver.maxGroupSize=maxGroupSize
        solver.run()
        
        solver = ALS(coeffs2, augmented_train_measures, train_values2,_verbosity=1)
        solver.maxSweeps = maxSweeps
        solver.targetResidual = 1e-10
        solver.maxGroupSize=maxGroupSize
        solver.method = 'l2'
        solver.run()
        
        testSampleSize = int(5e3)
        test_points = 2*np.random.rand(testSampleSize,order)-1
        test_values =  df(test_points)
        test_measures,test_measures_grad = legendre_measures_grad2(test_points, degree)
        #test_measures,test_measures_grad = monomial_measures_grad2(test_points, degree)
        augmented_test_measures =  \
            np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)
        augmented_test_measures_grad =  \
            np.concatenate([test_measures_grad, np.ones((1,testSampleSize,degree+1))], axis=0)
        
        
        def res(bstt, measure, measure_grad,values):
            res = 0
            for pos in range(bstt.order-1):
                tmp_measures = measure.copy()
                tmp_measures[pos] = measure_grad[pos]
                tmp_res = bstt.evaluate(tmp_measures)
                res += np.linalg.norm(tmp_res -  values[:,pos])**2 
            return np.sqrt(res) / np.linalg.norm(values)
        
        print("L2 grad: ",res(coeffs,augmented_test_measures,augmented_test_measures_grad,test_values)," on training data: ",res(coeffs,augmented_train_measures,augmented_train_measures_grad,train_values))
        print("L2 grad: ",res(coeffs2,augmented_test_measures,augmented_test_measures_grad,test_values)," on training data: ",res(coeffs2,augmented_train_measures,augmented_train_measures_grad,train_values))
   
        
        testSampleSize = int(2e4)
        test_points = 2*np.random.rand(testSampleSize,order)-1
        test_values = f(test_points)
        test_measures,test_measures_grad = legendre_measures_grad2(test_points, degree)
        #test_measures,test_measures_grad = monomial_measures_grad2(test_points, degree)
        augmented_test_measures =  \
            np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0) 
        values = coeffs.evaluate(augmented_test_measures)
        values2 = coeffs.evaluate(augmented_train_measures)
        a = legendre_measures(np.array([[0.0]*order]), degree)
        a = np.concatenate([a, np.ones((1,1,degree+1))], axis=0) 
        c = coeffs.evaluate(a)[0]-(order - 1)
        print("L2 scalar: ",np.linalg.norm(values-c -  test_values) / np.linalg.norm(test_values)," on training data: ",np.linalg.norm(values2-c -  train_values2) / np.linalg.norm(train_values2))
        
        res_tmp.append(np.linalg.norm(values-c -  test_values) / np.linalg.norm(test_values))

        values = coeffs2.evaluate(augmented_test_measures)
        values2 = coeffs2.evaluate(augmented_train_measures)
        print("L2 scalar: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)," on training data: ",np.linalg.norm(values2 -  train_values2) / np.linalg.norm(train_values2))
    result.append(res_tmp)