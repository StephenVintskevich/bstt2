#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:25:27 2021

@author: goette
"""
from math import comb
import numpy as np 
from misc import  __block, random_homogenous_polynomial_sum_system,zeros_homogenous_polynomial_sum_system,monomial_measures,legendre_measures,Gramian, HkinnerLegendre,random_full_system
from helpers import fermi_pasta_ulam
from als_l1_test import ALSSystem2
from bstt import Block, BlockSparseTT,BlockSparseTTSystem2
block = __block()

import warnings
warnings.filterwarnings("ignore")

order = 6
degree = 3
maxGroupSize = 2
interaction = 5
ranks = [4,4,4,4]
trainSampleSize = 10
maxSweeps=20
eq = 3

S = np.zeros([order,order+1])
for eq in range(order):
    for pos in range(order+1):
        if pos < eq-1:
            S[eq,pos] = 4
        elif pos == eq-1:
            S[eq,pos] = 3
        elif pos == eq:
            S[eq,pos] = 2
        elif pos == eq+1 and pos != order:
            S[eq,pos] = 1
        else:
            S[eq,pos] = 0
print(S)


train_points,train_values = fermi_pasta_ulam(order,trainSampleSize)
train_points = train_points.T
train_values = train_values.T
print(train_points.shape)
print(train_values.shape)
train_measures = monomial_measures(train_points, degree)
print(train_measures.shape)
augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)

def random_homogenous_polynomial_sum(_univariateDegrees, _totalDegree, _maxGroupSize):
    _univariateDegrees = np.asarray(_univariateDegrees, dtype=int)
    assert isinstance(_totalDegree, int) and _totalDegree >= 0
    assert _univariateDegrees.ndim == 1 and np.all(_univariateDegrees >= _totalDegree)
    order = len(_univariateDegrees)


    if isinstance(_maxGroupSize, int):
        _maxGroupSize = [_maxGroupSize]*len(_univariateDegrees)   

    def MaxSize(r,k):
        mr, mk = _totalDegree-r, order-k-1
        return min(comb(k+r,k), comb(mk+mr, mk), _maxGroupSize[k] if r == 1 else 1)

    dimensions = _univariateDegrees+1
    blocks = [[block[0,l,l] for l in range(_totalDegree+1)]]  # _totalDegree <= _univariateDegrees[0]
    ranks = []
    for k in range(1, order):
        mblocks = []
        leftSizes = [MaxSize(l,k-1) for l in range(_totalDegree+1)]
        leftSlices = np.cumsum([0] + leftSizes).tolist()
        rightSizes = [MaxSize(r,k) for r in range(_totalDegree+1)]
        rightSlices = np.cumsum([0] + rightSizes).tolist()
        for l in range(_totalDegree+1):
            for r in range(l, _totalDegree+1):  # If a polynomial of degree l is multiplied with another polynomial the degree must be at least l.
                m = r-l  # 0 <= m <= _totalDegree-l <= _totalDegree <= _univariateDegrees[m]
                mblocks.append(block[leftSlices[l]:leftSlices[l+1], m, rightSlices[r]:rightSlices[r+1]])
        ranks.append(leftSlices[-1])
        blocks.append(mblocks)
    #ranks.append(_totalDegree+1)
    #blocks.append([block[l,d-l,d] for d in range(_totalDegree+1) for l in range(d+1)])  # l+m == d <--> m == d-l
    ranks.append(_totalDegree+1)
    blocks.append([block[d,_totalDegree-d,0] for d in range(_totalDegree+1)])
    return BlockSparseTT.random(dimensions.tolist()+[_totalDegree+1], ranks, blocks)

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

blocks = random_homogenous_polynomial_sum([degree]*order, degree, maxGroupSize).blocks

print(blocks)
bstt_ex = BlockSparseTTSystem2(bstts,S,order)

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
exit
# bstt_test = BlockSparseTTSystem.random([degree+1]*(order+1), bstt_ex.ranks,bstt_ex.interaction,  blocks, order,selectionMatrix)
# bstt_test = bstt_ex
# #bstt_test = random_homogenous_polynomial_sum_system([degree]*order,interaction,degree,maxGroupSize,selectionMatrix)
# print(bstt_test.ranks)
# print(bstt_test.interaction)
# #bstt = random_full_system([degree]*order,interaction,ranks)
# print(f"DOFS: {bstt_test.dofs()}")
# localH1Gramians = [Gramian(degree+1,HkinnerLegendre(1)) for i in range(order)]
# localH1Gramians.append(np.ones([degree+1,degree+1]))
# localL2Gramians = [np.eye(degree+1) for i in range(order)]
# localL2Gramians.append(np.ones([degree+1,degree+1]))

# # #solver = ALSSystem(bstt_test, train_measures,  train_values,_localL2Gramians=localL2Gramians,_localH1Gramians=localH1Gramians,_verbosity=1)
# solver = ALSSystem(bstt_test, augmented_train_measures,  train_values,_verbosity=2)
# solver.maxSweeps = maxSweeps
# solver.targetResidual = 2e-8
# #solver.increaseRanks=increaseRanks
# solver.maxGroupSize=maxGroupSize
# solver.run()

# # for i in range(4**7):
# #     meas = np.zeros([7,1,4])
# #     a = [i //(4**(j-1)) % 4 for j in range(1,8)]
# #     meas[0,0,a[0]] = 1
# #     meas[1,0,a[1]] = 1
# #     meas[2,0,a[2]] = 1
# #     meas[3,0,a[3]] = 1
# #     meas[4,0,a[4]] = 1
# #     meas[5,0,a[5]] = 1
# #     meas[6,0,a[6]] = 1
# #     val = bstt_ex.evaluate(meas)
# #     val2 = bstt_test.evaluate(meas)
# #     if np.linalg.norm(val) != 0:
# #         print(a,": ",val,", ",val2)


# testSampleSize = int(1e5)
# test_points,test_values =fermi_pasta_ulam(order,testSampleSize)
# test_points = test_points.T
# test_values = test_values.T
# test_measures = monomial_measures(test_points, degree)
# augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)


# values = bstt_test.evaluate(augmented_test_measures)
# values2 = bstt_test.evaluate(augmented_train_measures)
# print("L1 sparse: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)," on training data: ",np.linalg.norm(values2 -  train_values) / np.linalg.norm(train_values))
