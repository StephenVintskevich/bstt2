import numpy as np

from misc import random_homogenous_polynomial_sum,  legendre_measures,legendre_measures_grad, Gramian, HkinnerLegendre  #, hermite_measures
from als import ALS




numSteps = 200
numTrials = 200
maxSweeps = 20
maxIter = 100

order = 32
degree = 5
maxGroupSize = 2  #NOTE: This is the block size needed to represent an arbitrary polynomial.



    
trainSampleSize = int(4e2)
testSampleSize = int(1e4)
train_points = 2*np.random.rand(trainSampleSize,order)-1
train_measures = legendre_measures(train_points, degree)
augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)

train_measures_grad =  legendre_measures_grad(train_points, degree) 
augmented_train_measures = [np.concatenate([train_measures_grad(k), np.ones((1,trainSampleSize,degree+1))], axis=0) for k in range(order)] 


train_values = f(train_points)


bstt = random_homogenous_polynomial_sum([degree]*order, degree, maxGroupSize)
localH1Gramians = [Gramian(degree+1,HkinnerLegendre(1)) for i in range(order)]
localH1Gramians.append(np.ones([degree+1,degree+1]))
localL2Gramians = [np.eye(degree+1) for i in range(order)]
localL2Gramians.append(np.ones([degree+1,degree+1]))
solver = ALS(bstt, augmented_train_measures,  train_values,_localL2Gramians=localL2Gramians,_localH1Gramians=localH1Gramians,_verbosity=1)
solver.maxSweeps = maxSweeps
solver.targetResidual = 1e-8
solver.run()