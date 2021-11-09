import os
from math import comb, factorial
from itertools import product

import numpy as np

from misc import random_homogenous_polynomial_sum,  legendre_measures, Gramian, HkinnerLegendre  #, hermite_measures
from als import ALS
from als_old import ALS as ALS_OLD




numSteps = 200
numTrials = 200
maxSweeps = 20
maxIter = 100

order = 10
degree = 7
maxGroupSize = 2  #NOTE: This is the block size needed to represent an arbitrary polynomial.


bandSize = 6
A = np.zeros([order,order])
A += np.diag(np.random.rand(order))
eps = 1
for i in range(1,bandSize):
    eps/=5
    A += eps*np.diag(np.random.rand(order-i),-i)
    A += eps*np.diag(np.random.rand(order-i),i)
    
print(np.around(A,4))
f = lambda xs: np.einsum('jk,ij,ik->i',A,xs,xs)  #NOTE: This functions gets peakier for larger M!
sampleSizes = np.unique(np.geomspace(1e1, 1e6, numSteps).astype(int))
trainSampleSize = int(4e2)
testSampleSize = int(1e4)


train_points = 2*np.random.rand(trainSampleSize,order)-1
train_measures = legendre_measures(train_points, degree)
augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)
train_values = f(train_points)
print(train_values.shape)
bstt_sum = random_homogenous_polynomial_sum([degree]*order, degree, maxGroupSize)
print(bstt_sum.dofs())
localH1Gramians = [Gramian(degree+1,HkinnerLegendre(1)) for i in range(order)]
localH1Gramians.append(np.ones([degree+1,degree+1]))
localL2Gramians = [np.eye(degree+1) for i in range(order)]
localL2Gramians.append(np.ones([degree+1,degree+1]))
solver = ALS(bstt_sum, augmented_train_measures,  train_values,_localL2Gramians=localL2Gramians,_localH1Gramians=localH1Gramians,_verbosity=1)
solver.maxSweeps = maxSweeps
solver.targetResidual = 1e-8
solver.run()

bstt_sum2 = random_homogenous_polynomial_sum([degree]*order, degree, maxGroupSize)
solver = ALS_OLD(bstt_sum2, augmented_train_measures,  train_values,_verbosity=1)
solver.maxSweeps = maxSweeps
solver.targetResidual = 1e-8
solver.run()

test_points = 2*np.random.rand(testSampleSize,order)-1
test_measures = legendre_measures(test_points, degree)
augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)
test_values = f(test_points)


values = bstt_sum.evaluate(augmented_test_measures)
print("L1: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values))

values = bstt_sum2.evaluate(augmented_test_measures)
print("L2: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values))