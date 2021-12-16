#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:25:27 2021

@author: goette
"""
import numpy as np 
from misc import  __block, sinecosine_measures, random_fixed_variable_sum_system,random_fixed_variable_sum,random_homogenous_polynomial_sum_system,random_homogenous_polynomial_sum,zeros_homogenous_polynomial_sum_system,monomial_measures,legendre_measures,Gramian, HkinnerLegendre,random_full_system
from helpers import magneticDipolesSamples, selectionMatrix1 as SM
from als_l1_test import ALSSystem
from bstt import BlockSparseTT
block = __block()

import warnings
warnings.filterwarnings("ignore")

order = 20
degree = 2
maxGroupSize = 3
#interaction = [4]+ [5] + [6]+[7]*(order-6) +[6]+[5] + [4]
#interaction = [3]+ [4] + [5]*(order-4) +[4] + [3]
#interaction = [3]+  [3] + [4]*(order-4) +[3] + [3]
interaction = [2]+   [3]*(order-2)  + [2]
trainSampleSize = 20000
maxSweeps=10
ranks = [4]*(order-1)


M = np.ones(order)
I = np.ones(order)
x = np.linspace(0,1*(order-1),order)




train_points,train_values = magneticDipolesSamples(order,trainSampleSize,M,x,I)
train_points = train_points.T
train_values = train_values.T
print(train_points.shape,np.linalg.norm(train_points))
print(train_values.shape,np.linalg.norm(train_values))
#train_measures = legendre_measures(train_points, degree,-np.pi,np.pi)
train_measures = sinecosine_measures(train_points)
print(train_measures.shape)
augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)

#bstt = random_homogenous_polynomial_sum_system([degree]*order,interaction,degree,maxGroupSize,selectionMatrix3)
bstt = random_fixed_variable_sum_system([degree]*order,interaction,degree,maxGroupSize,SM)
print(f"DOFS: {bstt.dofs()}")
print(f"Ranks: {bstt.ranks}")
print(f"Interaction: {bstt.interaction}")

   
solver = ALSSystem(bstt, augmented_train_measures,  train_values,_verbosity=1)
solver.maxSweeps = maxSweeps
solver.targetResidual = 1e-3
solver.maxGroupSize=maxGroupSize
solver.run()

testSampleSize = int(2e4)
test_points,test_values = magneticDipolesSamples(order,testSampleSize,M,x,I)
test_points = test_points.T
test_values = test_values.T
#test_measures = legendre_measures(test_points, degree,-np.pi,np.pi)
test_measures = sinecosine_measures(test_points)
augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)


values = bstt.evaluate(augmented_test_measures)
values2 = bstt.evaluate(augmented_train_measures)
print("L2: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)," on training data: ",np.linalg.norm(values2 -  train_values) / np.linalg.norm(train_values))

ret = np.ones((1,1,order))
for pos in range(bstt.order):
    comp = bstt.components[pos]
    S = SM(pos,order)
    ret = np.einsum('ijd,ipmk,jpnl,md,nd -> kld', ret, comp,comp, S,S)
assert ret.shape == (1,1,order)
print(ret[0,0,:])

comps = []
pos = 6
for k in range(pos):
    comps.append(bstt.components[k][:,:,0,:])
comps.append(bstt.components[pos][:,:,1,:])
for k in range(pos+1,order-1):
    comps.append(bstt.components[k][:,:,2,:])
comps.append(bstt.components[order -1][:,:,1,:])
comps.append(bstt.components[order][:,:,0,:])

tmp = random_fixed_variable_sum([degree]*order,degree,maxGroupSize)
bstt6 = BlockSparseTT(comps,tmp.blocks)

values = bstt6.evaluate(augmented_test_measures)
print("L2: ",np.linalg.norm(values -  test_values[:,pos]) / np.linalg.norm(test_values[:,pos]))
print(values[0]-test_values[0,pos])
for c in comps:
    print(np.linalg.norm(c),c.shape)
ret = np.ones((1,1))   
for pos in range(bstt6.order):
    comp = bstt6.components[pos]
    ret = np.einsum('ij,ipk,jpl -> kl', ret, comp,comp)
print(ret[0,0])
