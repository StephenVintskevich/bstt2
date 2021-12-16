#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:25:27 2021

@author: goette
"""
import numpy as np 
from misc import  __block, random_homogenous_polynomial_sum_system,zeros_homogenous_polynomial_sum_system,monomial_measures,legendre_measures,Gramian, HkinnerLegendre,random_full_system
from helpers import fermi_pasta_ulam
from als_l1_test import ALSSystem
from bstt import Block, BlockSparseTT,BlockSparseTTSystem
block = __block()

import warnings
warnings.filterwarnings("ignore")

order = 6
degree = 3
maxGroupSize = 2
interaction = [3]+ [4] + [5]*(order-4) + [4] + [3]
ranks = [4,4,4,4]
trainSampleSize = 10
maxSweeps=20
eq = 3


def selectionMatrix(k,_numberOfEquations):
    assert k >= 0 and k < _numberOfEquations+1
    if k == 0:
        Smat = np.zeros([3,_numberOfEquations])
        Smat[2,0] = 1
        Smat[1,1] = 1
        for i in range(2,_numberOfEquations):
            Smat[0,i] = 1
    elif k == 1:
        Smat = np.zeros([4,_numberOfEquations])
        Smat[3,0] = 1
        Smat[2,1] = 1
        Smat[1,2] = 1
        for i in range(3,_numberOfEquations):
            Smat[0,i] = 1
    elif k == _numberOfEquations - 2:
        Smat = np.zeros([4,_numberOfEquations])
        for i in range(0,k-1):
            Smat[3,i] = 1
        Smat[2,k-1] = 1
        Smat[1,k] = 1
        Smat[0,k+1] = 1
    elif k == _numberOfEquations - 1:
        Smat = np.zeros([3,_numberOfEquations])
        for i in range(0,k-1):
            Smat[2,i] = 1
        Smat[1,k-1] = 1
        Smat[0,k] = 1
    elif k == _numberOfEquations:
         Smat = np.zeros([1,_numberOfEquations])
         for i in range(0,_numberOfEquations):
             Smat[0,i] = 1
    else:        
        Smat = np.zeros([5,_numberOfEquations])
        for i in range(0,k-1):
            Smat[4,i] = 1
        Smat[3,k-1] = 1
        Smat[2,k] = 1
        Smat[1,k+1] = 1
        for i in range(k+2,_numberOfEquations):
            Smat[0,i] = 1
    return Smat

train_points,train_values = fermi_pasta_ulam(order,trainSampleSize)
train_points = train_points.T
train_values = train_values.T
print(train_points.shape)
print(train_values.shape)
train_measures = monomial_measures(train_points, degree)
print(train_measures.shape)
augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)


comps = []
blocks = []
b = [block[0:1,0:1,0:3,0:1],block[0:1,1:2,0:3,1:2],block[0:1,2:3,0:3,2:3],block[0:1,3:4,0:3,3:4]]
c = np.zeros([1,4,3,4])
c[0,0,0,0] = 1

c[0,0,1,0] = 1
c[0,1,1,1] = 1
c[0,2,1,2] = 1
c[0,3,1,3] = 1

c[0,0,2,0] = 1
c[0,1,2,1] = 1
#c[0,1,2,1] = -2
#c[0,1,2,2] = -3*0.7
c[0,2,2,2] = 3*0.7
c[0,3,2,3] = -2*0.7
comps.append(c)
blocks.append(b)
b = [block[0:1,0:1,0:4,0:1],block[0:1,1:2,0:4,1:3],block[0:1,2:3,0:4,3:4],block[0:1,3:4,0:4,4:5],block[1:2,0:1,0:4,1:3],block[1:2,1:2,0:4,3:4],block[1:2,2:3,0:4,4:5],block[2:3,0:1,0:4,3:4],block[2:3,1:2,0:4,4:5],block[3:4,0:1,0:4,4:5]]
c = np.zeros([4,4,4,5])
c[0,0,0,0] = 1

c[0,0,1,0] = 1
c[0,1,1,1] = 1
c[0,2,1,3] = 1
c[0,3,1,4] = 1

c[0,0,2,0] = 1
c[0,1,2,1] = -2
c[0,1,2,2] = -3*0.7
c[0,2,2,3] = 3*0.7
c[0,3,2,4] = -2*0.7
c[1,0,2,1] = 1
c[1,2,2,4] = 3*0.7
c[2,1,2,4] = -3*0.7
c[3,0,2,4] = 0.7

c[0,1,3,1] = 1
c[0,3,3,4] = 0.7
c[1,0,3,1] = -2
c[1,2,3,4] = -3*0.7
c[2,1,3,4] = 1
c[3,0,3,4] = 1
comps.append(c)
blocks.append(b)
for k in range(1,order+1-4):
    c = np.zeros([5,4,5,5])
    b = [block[0:1,0:1,0:5,0:1],block[0:1,1:2,0:5,1:3],block[0:1,2:3,0:5,3:4],block[0:1,3:4,0:5,4:5],block[1:3,0:1,0:5,1:3],block[1:3,1:2,0:5,3:4],block[1:3,2:3,0:5,4:5],block[3:4,0:1,0:5,3:4],block[3:4,1:2,0:5,4:5],block[4:5,0:1,0:5,4:5]]
    c[0,0,0,0] = 1
    
    c[0,0,1,0] = 1
    c[0,1,1,1] = 1
    c[0,2,1,3] = 1
    c[0,3,1,4] = 1
    
    c[0,0,2,0] = 1
    c[0,1,2,1] = -2
    c[0,1,2,2] = -3*0.7
    c[0,2,2,3] = 3*0.7
    c[0,3,2,4] = -2*0.7
    c[1,0,2,1] = 1
    c[1,2,2,4] = 3*0.7
    c[3,1,2,4] = -3*0.7
    c[4,0,2,4] = 0.7
    
    c[0,1,3,1] = 1
    c[0,3,3,4] = 0.7
    c[1,0,3,1] = 1
    c[2,2,3,4] = 1
    c[3,1,3,4] = 1
    c[4,0,3,4] = 1
    
    c[1,0,4,1] = 1
    c[4,0,4,4] = 1
    comps.append(c)
    blocks.append(b)

c = np.zeros([5,4,4,5])
b = [block[0:1,0:1,0:4,0:1],block[0:1,1:2,0:4,1:3],block[0:1,2:3,0:4,3:4],block[0:1,3:4,0:4,4:5],block[1:3,0:1,0:4,1:3],block[1:3,1:2,0:4,3:4],block[1:3,2:3,0:4,4:5],block[3:4,0:1,0:4,3:4],block[3:4,1:2,0:4,4:5],block[4:5,0:1,0:4,4:5]]
c[0,0,0,0] = 1
c[0,1,0,1] = 1
c[0,2,0,3] = 1
c[0,3,0,4] = 1

c[0,0,1,0] = 1
c[0,1,1,1] = -2
c[0,1,1,2] = -3*0.7
c[0,2,1,3] = 3*0.7
c[0,3,1,4] = -2*0.7
c[1,0,1,1] = 1
c[1,2,1,4] = 3*0.7
c[3,1,1,4] = -3*0.7
c[4,0,1,4] = 0.7

c[0,1,2,1] = 1
c[0,3,2,4] = 0.7
c[1,0,2,1] = 1
c[2,2,2,4] = 1
c[3,1,2,4] = 1
c[4,0,2,4] = 1

c[1,0,3,1] = 1
c[4,0,3,4] = 1
comps.append(c)
blocks.append(b)

c = np.zeros([5,4,3,4])
b = [block[0:1,0:1,0:3,0:1],block[0:1,1:2,0:3,1:2],block[0:1,2:3,0:3,2:3],block[0:1,3:4,0:3,3:4],block[1:3,0:1,0:3,1:2],block[1:3,1:2,0:3,2:3],block[1:3,2:3,0:3,3:4],block[3:4,0:1,0:3,2:3],block[3:4,1:2,0:3,3:4],block[4:5,0:1,0:3,3:4]]
 
 
#c[0,0,0,0] = 1

c[0,1,0,1] = -2
c[0,3,0,3] = -2*0.7
c[1,0,0,1] = 1
c[1,2,0,3] = 3*0.7
c[3,1,0,3] = -3*0.7
c[4,0,0,3] = 0.7

c[0,1,1,1] = 1
c[0,3,1,3] = 0.7
c[1,0,1,1] = 1
c[2,2,1,3] = 1
c[3,1,1,3] = 1
c[4,0,1,3] = 1

c[1,0,2,1] = 1
c[4,0,2,3] = 1
comps.append(c)
blocks.append(b)

c = np.zeros([4,4,1,1])
b = [block[0:1,3:4,0:1,0:1],block[1:2,2:3,0:1,0:1],block[2:3,1:2,0:1,0:1],block[3:4,0:1,0:1,0:1]]
c[3,0,0,0] = 1
c[2,1,0,0] = 1
c[1,2,0,0] = 1
c[0,3,0,0] = 1
comps.append(c)
blocks.append(b)
print(blocks)
bstt_ex = BlockSparseTTSystem(comps,blocks,selectionMatrix,_numberOfEquations=order)

core = comps[3]
sh = core.shape
ctmp = np.einsum('ijkl->ijlk',core).reshape(sh[0]*sh[1]*sh[3],sh[2])
U,s,VT = np.linalg.svd(ctmp)
print("sing val: ", s)


bstt_ex.assume_corePosition(bstt_ex.order-1)
while bstt_ex.corePosition > 0:
     bstt_ex.move_core('left')

print(f"Norm: {np.linalg.norm(bstt_ex.components[0])}")
print(bstt_ex.ranks)
print(bstt_ex.interaction)
print(augmented_train_measures.shape)
print(train_values.shape)
print(f"order: {bstt_ex.order}")
values = bstt_ex.evaluate(augmented_train_measures)
print("Error: ",np.linalg.norm(values -  train_values) / np.linalg.norm(train_values))
print(augmented_train_measures.shape)

for i in range(4**7):
    meas = np.zeros([7,1,4])
    a = [i //(4**(j-1)) % 4 for j in range(1,8)]
    meas[0,0,a[0]] = 1
    meas[1,0,a[1]] = 1
    meas[2,0,a[2]] = 1
    meas[3,0,a[3]] = 1
    meas[4,0,a[4]] = 1
    meas[5,0,a[5]] = 1
    meas[6,0,a[6]] = 1
    val = bstt_ex.evaluate(meas)
    if np.linalg.norm(val) != 0:
        print(a,": ",val)

bstt_test = BlockSparseTTSystem.random([degree+1]*(order+1), bstt_ex.ranks,bstt_ex.interaction,  blocks, order,selectionMatrix)
bstt_test = bstt_ex
#bstt_test = random_homogenous_polynomial_sum_system([degree]*order,interaction,degree,maxGroupSize,selectionMatrix)
print(bstt_test.ranks)
print(bstt_test.interaction)
#bstt = random_full_system([degree]*order,interaction,ranks)
print(f"DOFS: {bstt_test.dofs()}")
localH1Gramians = [Gramian(degree+1,HkinnerLegendre(1)) for i in range(order)]
localH1Gramians.append(np.ones([degree+1,degree+1]))
localL2Gramians = [np.eye(degree+1) for i in range(order)]
localL2Gramians.append(np.ones([degree+1,degree+1]))

# #solver = ALSSystem(bstt_test, train_measures,  train_values,_localL2Gramians=localL2Gramians,_localH1Gramians=localH1Gramians,_verbosity=1)
solver = ALSSystem(bstt_test, augmented_train_measures,  train_values,_verbosity=2)
solver.maxSweeps = maxSweeps
solver.targetResidual = 2e-8
#solver.increaseRanks=increaseRanks
solver.maxGroupSize=maxGroupSize
solver.run()

# for i in range(4**7):
#     meas = np.zeros([7,1,4])
#     a = [i //(4**(j-1)) % 4 for j in range(1,8)]
#     meas[0,0,a[0]] = 1
#     meas[1,0,a[1]] = 1
#     meas[2,0,a[2]] = 1
#     meas[3,0,a[3]] = 1
#     meas[4,0,a[4]] = 1
#     meas[5,0,a[5]] = 1
#     meas[6,0,a[6]] = 1
#     val = bstt_ex.evaluate(meas)
#     val2 = bstt_test.evaluate(meas)
#     if np.linalg.norm(val) != 0:
#         print(a,": ",val,", ",val2)


testSampleSize = int(1e5)
test_points,test_values =fermi_pasta_ulam(order,testSampleSize)
test_points = test_points.T
test_values = test_values.T
test_measures = monomial_measures(test_points, degree)
augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)


values = bstt_test.evaluate(augmented_test_measures)
values2 = bstt_test.evaluate(augmented_train_measures)
print("L1 sparse: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)," on training data: ",np.linalg.norm(values2 -  train_values) / np.linalg.norm(train_values))
