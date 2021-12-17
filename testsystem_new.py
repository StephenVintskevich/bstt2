#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:25:27 2021

@author: goette
"""
from math import comb
import numpy as np 
from misc import  __block,legendre_measures,Gramian, HkinnerLegendre
from helpers import fermi_pasta_ulam,selectionMatrix1 as SM
from als_l1_test import ALSSystem2
from bstt import BlockSparseTTSystem2,BlockSparseTT,BlockSparseTensor
block = __block()

import warnings
warnings.filterwarnings("ignore")

order = 6 
degree = 3
maxGroupSize = 2
interaction = 5

trainSampleSize = 10000
maxSweeps=20
ranks = [4]*(order-1)

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
# =============================================================================
# S = np.zeros([order,order+1])
# for eq in range(order):
#     for pos in range(order+1):
#         if pos < eq:
#             S[eq,pos] = 2
#         elif pos == eq:
#             S[eq,pos] = 1
#         else:
#             S[eq,pos] = 0
# =============================================================================
print(S)

def random_homogenous_polynomial_sum_system2(_univariateDegrees, _totalDegree, _maxGroupSize,_numberOfInteractions,_selectionMatrix):
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
    numberOfEquations = len(dimensions)
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
    return BlockSparseTTSystem2.random(dimensions.tolist()+[_totalDegree+1], ranks, blocks,numberOfEquations,_numberOfInteractions,_selectionMatrix)


train_points,train_values = fermi_pasta_ulam(order,trainSampleSize)
train_points = train_points.T
train_values = train_values.T
train_measures = legendre_measures(train_points, degree)
augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)

coeffs = random_homogenous_polynomial_sum_system2([degree]*order,degree,maxGroupSize,interaction,S)
print(f"DOFS: {coeffs.dofs()}")
print(f"Ranks: {coeffs.ranks}")
print(f"Interaction: {coeffs.interactions}")



   
#solver = ALSSystem(bstt, train_measures,  train_values,_verbosity=1)
solver = ALSSystem2(coeffs, augmented_train_measures,  train_values,_verbosity=2)
solver.maxSweeps = maxSweeps
solver.targetResidual = 1e-4
#solver.increaseRanks=increaseRanks
solver.maxGroupSize=maxGroupSize
solver.run()

testSampleSize = int(2e4)
test_points,test_values =fermi_pasta_ulam(order,testSampleSize)
test_points = test_points.T
test_values = test_values.T
test_measures = legendre_measures(test_points, degree)
augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)


values = coeffs.evaluate(augmented_test_measures)
values2 = coeffs.evaluate(augmented_train_measures)
print("L2: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)," on training data: ",np.linalg.norm(values2 -  train_values) / np.linalg.norm(train_values))





