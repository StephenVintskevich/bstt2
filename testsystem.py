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
from als_l1_test import ALSSystem
from bstt import BlockSparseTTSystem,BlockSparseTT,BlockSparseTensor
block = __block()

import warnings
warnings.filterwarnings("ignore")

order = 20  
degree = 3
maxGroupSize = 2
#interaction = [3]+ [4] + [5]*(order-4) + [4] + [3]
interaction = [2]+   [3]*(order-2)  + [2]
#interaction = [2]*order

trainSampleSize = 10000
maxSweeps=20
ranks = [4]*(order-1)

def random_homogenous_polynomial_sum_system(_univariateDegrees, _interactionranges, _totalDegree, _maxGroupSize,_selectionMatrix):
    _univariateDegrees = np.asarray(_univariateDegrees, dtype=int)
    assert isinstance(_totalDegree, int) and _totalDegree >= 0
    assert _univariateDegrees.ndim == 1 and np.all(_univariateDegrees >= _totalDegree)
    assert len(_univariateDegrees) == len(_interactionranges)
    assert isinstance(_interactionranges,list)
    order = len(_univariateDegrees)
    
    
    

    def MaxSize(r,k):
        mr, mk = _totalDegree-r, order-k-1
        return min(comb(k+r,k), comb(mk+mr, mk), _maxGroupSize if r == 1 else 1)

    dimensions = _univariateDegrees+1
    numberOfEquations = len(dimensions)

    blocks = [[block[0,l,0:_interactionranges[0],l] for l in range(_totalDegree+1)]]  # _totalDegree <= _univariateDegrees[0]
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
                mblocks.append(block[leftSlices[l]:leftSlices[l+1], m, 0:_interactionranges[k], rightSlices[r]:rightSlices[r+1]])
        ranks.append(leftSlices[-1])
        blocks.append(mblocks)
    #ranks.append(_totalDegree+1)
    #blocks.append([block[l,d-l,0:_interactionranges[-1],d] for d in range(_totalDegree+1) for l in range(d+1)])  # l+m == d <--> m == d-l
    ranks.append(_totalDegree+1)
    blocks.append([block[d,_totalDegree-d,0,0] for d in range(_totalDegree+1)])
    return BlockSparseTTSystem.random(dimensions.tolist()+[_totalDegree+1], ranks,_interactionranges +[1],  blocks, numberOfEquations,_selectionMatrix)

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


train_points,train_values = fermi_pasta_ulam(order,trainSampleSize)
train_points = train_points.T
train_values = train_values.T
print(train_points.shape)
print(train_values.shape)
train_measures = legendre_measures(train_points, degree)
print(train_measures.shape)
augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)

bstt = random_homogenous_polynomial_sum_system([degree]*order,interaction,degree,maxGroupSize,SM)
#bstt = random_full_system([degree]*order, interaction2, ranks,selectionMatrix2)
print(f"DOFS: {bstt.dofs()}")
print(f"Ranks: {bstt.ranks}")
print(f"Interaction: {bstt.interaction}")
localH1Gramians = [Gramian(degree+1,HkinnerLegendre(1)) for i in range(order)]
localH1Gramians.append(np.ones([degree+1,degree+1]))
localL2Gramians = [np.eye(degree+1) for i in range(order)]
localL2Gramians.append(np.ones([degree+1,degree+1]))

   
#solver = ALSSystem(bstt, train_measures,  train_values,_verbosity=1)
solver = ALSSystem(bstt, augmented_train_measures,  train_values,_localL2Gramians=localL2Gramians,_localH1Gramians=localH1Gramians,_verbosity=1)
solver.maxSweeps = maxSweeps
solver.targetResidual = 1e-6
#solver.increaseRanks=increaseRanks
solver.maxGroupSize=maxGroupSize
solver.run()

testSampleSize = int(2e4)
test_points,test_values =fermi_pasta_ulam(order,testSampleSize)
test_points = test_points.T
test_values = test_values.T
test_measures = legendre_measures(test_points, degree)
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


tmp = random_homogenous_polynomial_sum([degree]*order,degree,maxGroupSize)
for m, (comp, compBlocks) in enumerate(zip(comps, tmp.blocks)):
    print(m)
    BlockSparseTensor.fromarray(comp, compBlocks)

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




