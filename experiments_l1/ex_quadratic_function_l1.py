import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import os
import numpy as np

from misc import random_homogenous_polynomial_sum,  legendre_measures, Gramian, HkinnerLegendre  #, hermite_measures
from als import ALS




order = 10
degree = 6
maxGroupSize = 4  #NOTE: This is the block size needed to represent an arbitrary polynomial.
increaseRanks = True
maxSweeps = 10

bandSize = 3
A = np.zeros([order,order])
A += np.diag(np.random.rand(order))
eps = 1.0
fac = 10.0
for i in range(1,bandSize):
    eps/=fac
    A += eps*np.diag(np.random.rand(order-i),-i)
    A += eps*np.diag(np.random.rand(order-i),i)
    
print(np.around(A,4))
f = lambda xs: np.einsum('jk,ij,ik->i',A,xs,xs)  #NOTE: This functions gets peakier for larger M!
trainSampleSize = int(5000)
testSampleSize = int(1e5)


train_points = 2*np.random.rand(trainSampleSize,order)-1
train_measures = legendre_measures(train_points, degree)
augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)
train_values = f(train_points)
print(train_values.shape)
bstt_sum = random_homogenous_polynomial_sum([degree]*order, degree, 1 if increaseRanks else maxGroupSize)
print(f"DOFS: {bstt_sum.dofs()}")
print(f"DOFS: {bstt_sum.ranks}")
localH1Gramians = [Gramian(degree+1,HkinnerLegendre(1)) for i in range(order)]
localH1Gramians.append(np.ones([degree+1,degree+1]))
localL2Gramians = [np.eye(degree+1) for i in range(order)]
localL2Gramians.append(np.ones([degree+1,degree+1]))

solver = ALS(bstt_sum, augmented_train_measures,  train_values,_localL2Gramians=localL2Gramians,_localH1Gramians=localH1Gramians,_verbosity=1)
solver.maxSweeps = maxSweeps
solver.targetResidual = 1e-5
solver.increaseRanks=increaseRanks
solver.maxGroupSize=maxGroupSize
solver.run()
print(f"DOFS: {bstt_sum.ranks}")


bstt_sum3 = random_homogenous_polynomial_sum([degree]*order, degree, 1 if increaseRanks else maxGroupSize)
print(f"DOFS: {bstt_sum3.dofs()}")
print(f"DOFS: {bstt_sum3.ranks}")
solver = ALS(bstt_sum3, augmented_train_measures,  train_values,_verbosity=1)
solver.maxSweeps = maxSweeps
solver.targetResidual = 1e-5
solver.increaseRanks=increaseRanks
solver.maxGroupSize=maxGroupSize
solver.run()
print(f"DOFS: {bstt_sum3.ranks}")


bstt_sum2 = random_homogenous_polynomial_sum([degree]*order, degree,  1 if increaseRanks else maxGroupSize)
print(f"DOFS: {bstt_sum2.dofs()}")
print(f"DOFS: {bstt_sum2.ranks}")
solver = ALS(bstt_sum2, augmented_train_measures,  train_values,_verbosity=1)
solver.increaseRanks=increaseRanks
solver.method = 'l2'
solver.maxSweeps = maxSweeps
solver.targetResidual = 1e-5
solver.maxGroupSize=maxGroupSize
solver.run()



test_points = 2*np.random.rand(testSampleSize,order)-1
test_measures = legendre_measures(test_points, degree)
augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)
test_values = f(test_points)


values = bstt_sum.evaluate(augmented_test_measures)
values2 = bstt_sum.evaluate(augmented_train_measures)
print("L1 sparse with reg: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)," on training data: ",np.linalg.norm(values2 -  train_values) / np.linalg.norm(train_values))

values = bstt_sum3.evaluate(augmented_test_measures)
values2 = bstt_sum3.evaluate(augmented_train_measures)
print("L1 sparse without reg: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)," on training data: ",np.linalg.norm(values2 -  train_values) / np.linalg.norm(train_values))

values = bstt_sum2.evaluate(augmented_test_measures)
values2 = bstt_sum2.evaluate(augmented_train_measures)
print("L2 sparse: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)," on training data: ",np.linalg.norm(values2 -  train_values) / np.linalg.norm(train_values))

