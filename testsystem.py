#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:25:27 2021

@author: goette
"""
import numpy as np 
from misc import random_homogenous_polynomial_sum_system,random_homogenous_polynomial_sum,legendre_measures

order = 8
degree = 2
maxGroupSize = 1
interaction = [3,4,4,4,4,4,4,3]
trainSampleSize = 1

tt = random_homogenous_polynomial_sum_system([degree]*order,interaction,degree,maxGroupSize)


print(tt.dimensions, tt.ranks, tt.interaction)

train_points = np.ones([trainSampleSize,order])
train_measures = legendre_measures(train_points, degree)
augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)

print(tt.evaluate(augmented_train_measures))

tt.assume_corePosition(tt.order-1)
while tt.corePosition > 0:
    tt.move_core('left')


ret = np.ones((1,1))
for pos in reversed(range(1,tt.order)):
    ret = np.einsum('nm,ikln,jklm -> ij', ret, tt.components[pos],tt.components[pos])
    print(ret.shape,"\n",np.around(ret,4),"\n",np.max(tt.components[pos]))

tt.move_core('right')
tt.move_core('right')

a = tt.getAllBlocksOfSlice(1, slice(1,2,None), 3)
u = np.zeros(tt.components[tt.corePosition-1][:,:,:,0].shape)
for blk in a:
    u[blk[0],blk[1],blk[2]] = np.random.rand(blk[0].stop-blk[0].start,blk[1].stop-blk[1].start,blk[2].stop-blk[2].start)

tt.increase_block(1,u,np.zeros(tt.components[tt.corePosition][0,:,:,:].shape),'left')

print(tt.blocks[tt.corePosition-1])